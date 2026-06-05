import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from video_preprocessor import VideoPreprocessor
import torchvision.transforms.functional as F


class VideoDataset(Dataset):
    def __init__(self, manifest_path, video_dir, split, maxFrame, imgSize, transform=None, cache_dir=None, mask_dir=None):

        manifest = pd.read_csv(manifest_path)

        self.video_split = manifest[manifest["split"] == split].reset_index(drop=True)

        self.split = split
        self.video_dir = video_dir
        self.transform = transform
        
        # Nuove variabili per la gestione opzionale della cache offline
        self.cache_dir = cache_dir
        self.mask_dir = mask_dir

        conteggi = self.video_split.iloc[:, 5].value_counts()
        soglia_media = conteggi.mean()
        self.classi_rare = conteggi[conteggi < soglia_media].index.tolist()

        self.preprocessor = VideoPreprocessor(maxFrame, imgSize)

        # ==========================
        # MAPPING ORIGINALE
        # ==========================

        labels_complete = manifest.iloc[:, 5].values

        original_class_names = sorted(pd.unique(labels_complete))

        self.class_to_idx = {
            class_name: idx
            for idx, class_name in enumerate(original_class_names)
        }

        self.idx_to_class = {
            idx: class_name
            for class_name, idx in self.class_to_idx.items()
        }

        # ==========================
        # MAPPING AZIONE GENERALE
        # ==========================
        # Qui trasformiamo le 9 classi originali in 5 classi azione.

        self.action_mapping = {
            "idle": "idle",
            "non-gioco": "non-gioco",
            "passaggio": "passaggio",

            "tiroDaDue0": "tiroDaDue",
            "tiroDaDue1": "tiroDaDue",

            "tiroDaTre0": "tiroDaTre",
            "tiroDaTre1": "tiroDaTre",

            "tiroLibero0": "tiroLibero",
            "tiroLibero1": "tiroLibero",
        }

        action_names_complete = [
            self.action_mapping[label_name]
            for label_name in labels_complete
        ]

        action_class_names = sorted(pd.unique(action_names_complete))

        self.action_to_idx = {
            action_name: idx
            for idx, action_name in enumerate(action_class_names)
        }

        self.idx_to_action = {
            idx: action_name
            for action_name, idx in self.action_to_idx.items()
        }

        # Mapping esito tiro
        self.outcome_to_idx = {
            "sbagliato": 0,
            "segnato": 1
        }

        self.idx_to_outcome = {
            0: "sbagliato",
            1: "segnato"
        }

    def __len__(self):
        return len(self.video_split)

    def __getitem__(self, idx):
        rel_path = self.video_split.iloc[idx, 1]
        label_name = self.video_split.iloc[idx, 5]

        video_path = os.path.normpath(
            os.path.join(self.video_dir, rel_path)
        )

        # ==========================
        # CARICAMENTO DATI (CACHE OFFLINE O PREPROCESSOR)
        # ==========================
        usato_cache = False

        if self.cache_dir and self.mask_dir:
            nome_file = rel_path.replace("/", "_").replace("\\", "_") + ".npy"
            nome_file_mask = rel_path.replace("/", "_").replace("\\", "_") + "_mask.npy"

            percorso_frames_cache = os.path.join(self.cache_dir, nome_file)
            percorso_mask_cache = os.path.join(self.mask_dir, nome_file_mask)

            if os.path.exists(percorso_frames_cache) and os.path.exists(percorso_mask_cache):
                frames = np.load(percorso_frames_cache)
                mask = np.load(percorso_mask_cache)
                usato_cache = True

        # Fallback se le cartelle di cache non sono fornite o i file .npy non esistono
        if not usato_cache:
            frames, mask, total_frames = self.preprocessor(video_path)

        # Trasforma l'array NumPy caricato in un torch.Tensor ed esegue il permute
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        mask = torch.from_numpy(mask).long()

        
        if self.transform:

            if self.split == "train":

                # 1. Flip orizzontale casuale (indipendente)
                if torch.rand(1).item() > 0.5:
                    frames = torch.stack([
                        F.hflip(f)
                        for f in frames
                    ])

                # 2. Luminosità casuale (allineata correttamente, ora è indipendente!)
                # Intervallo ottimizzato a (0.6, 1.4) come richiesto
                bright_factor = torch.empty(1).uniform_(0.6, 1.4).item()                
                frames = torch.stack([
                    F.adjust_brightness(f, bright_factor)
                    for f in frames
                ])

            # Normalizzazione MobileNet finale applicata frame per frame
            frames = torch.stack([
                self.transform(f)
                for f in frames
            ])   

        # ==========================
        # LABEL AZIONE
        # ==========================

        action_name = self.action_mapping[label_name]
        action_label = self.action_to_idx[action_name]

        # ==========================
        # LABEL CANESTRO / ESITO
        # ==========================

        if label_name in ["tiroDaDue0", "tiroDaTre0", "tiroLibero0"]:
            is_shot = True
            canestro = 0   # tiro sbagliato

        elif label_name in ["tiroDaDue1", "tiroDaTre1", "tiroLibero1"]:
            is_shot = True
            canestro = 1   # tiro segnato

        else:
            is_shot = False
            canestro = -1  # non ha senso per passaggio / idle / non-gioco

        # ==========================
        # CONVERSIONE IN TENSORI FINALE
        # ==========================

        action_label = torch.tensor(action_label, dtype=torch.long)

        # canestro deve essere long perché verrà usato con CrossEntropyLoss
        canestro = torch.tensor(canestro, dtype=torch.long)

        # is_shot deve essere bool perché serve come maschera
        is_shot = torch.tensor(is_shot, dtype=torch.bool)

        return frames, mask, action_label, canestro, is_shot