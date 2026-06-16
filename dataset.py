import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class VideoDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        video_dir,
        split,
        maxFrame,
        imgSize,
        transform=None,
        cache_dir=None,
        mask_dir=None,
        movinet_features_dir=None, # Non più strettamente necessario, ma mantenuto per compatibilità
        rfdetr_features_dir=None,
        rfdetr_feature_dim=19
    ):

        manifest = pd.read_csv(manifest_path)
        self.video_split = manifest[manifest["split"] == split].reset_index(drop=True)

        self.split = split
        self.video_dir = video_dir
        self.transform = transform
        self.maxFrame = maxFrame

        # Cache dei frame reali, delle mask e delle feature geometriche RF-DETR.
        self.cache_dir = cache_dir       # Cartella "video_32_frame"
        self.mask_dir = mask_dir         # Cartella "mask_frame"
        self.rfdetr_features_dir = rfdetr_features_dir
        self.rfdetr_feature_dim = rfdetr_feature_dim

        # Mapping delle azioni ed etichette ereditato dal tuo codice originale
        self.action_mapping = {
            "idle": "idle",
            "non-gioco": "non-gioco",
            "passaggio": "passaggio",
            "tiroDaDue0": "tiroDaDue",
            "tiroDaDue1": "tiroDaDue",
            "tiroDaTre0": "tiroDaTre",
            "tiroDaTre1": "tiroDaTre",
            "tiroLibero0": "tiroLibero",
            "tiroLibero1": "tiroLibero"
        }
        
        self.action_to_idx = {
            "idle": 0,
            "non-gioco": 1,
            "passaggio": 2,
            "tiroDaDue": 3,
            "tiroDaTre": 4,
            "tiroLibero": 5
        }

    def __len__(self):
        return len(self.video_split)

    def _load_rfdetr_features(self, nome_file):
        percorso_rfdetr = os.path.join(self.rfdetr_features_dir, nome_file)
        if os.path.exists(percorso_rfdetr):
            feat = np.load(percorso_rfdetr).astype(np.float32)
            if feat.shape[0] < self.maxFrame:
                pad_width = ((0, self.maxFrame - feat.shape[0]), (0, 0))
                feat = np.pad(feat, pad_width, mode='constant', constant_values=0)
            elif feat.shape[0] > self.maxFrame:
                feat = feat[:self.maxFrame, :]
            return feat
        else:
            return np.zeros((self.maxFrame, self.rfdetr_feature_dim), dtype=np.float32)

    def __getitem__(self, idx):
        row = self.video_split.iloc[idx]
        path_video = row["path"]
        label_name = row["label"]

        nome_file = path_video.replace("/", "_").replace("\\", "_") + ".npy"
        nome_file_mask = nome_file.replace(".npy", "_mask.npy")

        # ==========================================
        # 1. CARICAMENTO DEI FRAME VERI (Dal vivo)
        # ==========================================
        percorso_cache = os.path.join(self.cache_dir, nome_file)
        frames = np.load(percorso_cache)  # Shape nativa: [32, 704, 704, 3], uint8
        frames = torch.from_numpy(frames) # Tensore [T, H, W, C]

        # Data Augmentation "Live" eseguita solo nel subset di Training
        if self.split == "train":
            # Permutiamo temporaneamente in [T, C, H, W] per usare i moduli di torchvision
            frames = frames.permute(0, 3, 1, 2).float()
            
            # Flip orizzontale casuale
            if torch.rand(1).item() > 0.5:
                frames = torch.stack([F.hflip(f) for f in frames])

            # Variazione casuale della luminosità
            if torch.rand(1).item() > 0.5:
                bright_factor = torch.empty(1).uniform_(0.8, 1.2).item()
                frames = torch.stack([F.adjust_brightness(f, bright_factor) for f in frames])
            
            # Ritorniamo alla forma standard [T, H, W, C] in formato uint8 per risparmiare memoria
            frames = frames.permute(0, 2, 3, 1).to(torch.uint8)

        if self.transform:
            frames = torch.stack([self.transform(f) for f in frames])

        # ==========================================
        # 2. CARICAMENTO MASK TEMPORALE
        # ==========================================
        percorso_mask_cache = os.path.join(self.mask_dir, nome_file_mask)
        mask = np.load(percorso_mask_cache)
        mask = torch.from_numpy(mask).long().reshape(-1)

        # ==========================================
        # 3. CARICAMENTO FEATURE GEOMETRICHE RF-DETR
        # ==========================================
        rfdetr_features = self._load_rfdetr_features(nome_file)
        rfdetr_features = torch.from_numpy(rfdetr_features).float()

        # ==========================================
        # 4. GESTIONE DELLE ETICHETTE (AZIONI ED ESITI)
        # ==========================================
        action_name = self.action_mapping[label_name]
        action_label = self.action_to_idx[action_name]

        if label_name in ["tiroDaDue0", "tiroDaTre0", "tiroLibero0"]:
            is_shot = True
            canestro = 0
        elif label_name in ["tiroDaDue1", "tiroDaTre1", "tiroLibero1"]:
            is_shot = True
            canestro = 1
        else:
            is_shot = False
            canestro = -1

        action_label = torch.tensor(action_label, dtype=torch.long)
        canestro = torch.tensor(canestro, dtype=torch.long)
        is_shot = torch.tensor(is_shot, dtype=torch.bool)

        return frames, mask, rfdetr_features, action_label, canestro, is_shot