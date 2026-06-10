import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from video_preprocessor import VideoPreprocessor
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
        rfdetr_features_dir=None,
        rfdetr_feature_dim=19
    ):

        manifest = pd.read_csv(manifest_path)

        self.video_split = manifest[manifest["split"] == split].reset_index(drop=True)

        self.split = split
        self.video_dir = video_dir
        self.transform = transform
        self.maxFrame = maxFrame

        # Cache offline dei frame, delle mask e delle feature RF-DETR.
        self.cache_dir = cache_dir
        self.mask_dir = mask_dir
        self.rfdetr_features_dir = rfdetr_features_dir
        self.rfdetr_feature_dim = rfdetr_feature_dim

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

        # ==========================
        # MAPPING ESITO TIRO
        # ==========================

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

    def _cache_file_names(self, rel_path):
        nome_file = rel_path.replace("/", "_").replace("\\", "_") + ".npy"
        nome_file_mask = rel_path.replace("/", "_").replace("\\", "_") + "_mask.npy"
        return nome_file, nome_file_mask

    def _load_rfdetr_features(self, nome_file):
        """
        Carica feature RF-DETR già estratte.
        Se mancano, crea zeri [maxFrame, rfdetr_feature_dim].
        """

        if self.rfdetr_features_dir is None:
            return np.zeros(
                (self.maxFrame, self.rfdetr_feature_dim),
                dtype=np.float32
            )

        path_features = os.path.join(self.rfdetr_features_dir, nome_file)

        if not os.path.exists(path_features):
            print(f"ATTENZIONE: feature RF-DETR mancanti: {path_features}")
            return np.zeros(
                (self.maxFrame, self.rfdetr_feature_dim),
                dtype=np.float32
            )

        features = np.load(path_features).astype(np.float32)

        # Sicurezza forma.
        if features.ndim != 2:
            raise ValueError(
                f"Feature RF-DETR con shape non valida: {path_features}, shape={features.shape}"
            )

        # Se il numero frame non coincide, taglio o padding.
        if features.shape[0] > self.maxFrame:
            features = features[:self.maxFrame]

        elif features.shape[0] < self.maxFrame:
            padding = np.zeros(
                (self.maxFrame - features.shape[0], features.shape[1]),
                dtype=np.float32
            )
            features = np.concatenate([features, padding], axis=0)

        return features

    def __getitem__(self, idx):
        rel_path = self.video_split.iloc[idx, 1]
        label_name = self.video_split.iloc[idx, 5]

        video_path = os.path.normpath(
            os.path.join(self.video_dir, rel_path)
        )

        # ==========================
        # CARICAMENTO FRAME + MASK
        # ==========================

        usato_cache = False
        nome_file, nome_file_mask = self._cache_file_names(rel_path)

        if self.cache_dir and self.mask_dir:
            percorso_frames_cache = os.path.join(self.cache_dir, nome_file)
            percorso_mask_cache = os.path.join(self.mask_dir, nome_file_mask)

            if os.path.exists(percorso_frames_cache) and os.path.exists(percorso_mask_cache):
                frames = np.load(percorso_frames_cache)
                mask = np.load(percorso_mask_cache)
                usato_cache = True

        if not usato_cache:
            frames, mask, total_frames = self.preprocessor(video_path)

        # ==========================
        # CARICAMENTO FEATURE RF-DETR
        # ==========================

        rfdetr_features = self._load_rfdetr_features(nome_file)

        # ==========================
        # CONVERSIONE FRAME / MASK
        # ==========================

        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        mask = torch.from_numpy(mask).long().reshape(-1)

        rfdetr_features = torch.from_numpy(rfdetr_features).float()

        # ==========================
        # TRASFORMAZIONI OPZIONALI
        # ==========================
        # Nota:
        # se usi feature RF-DETR già estratte, è meglio lasciare transform=None.
        # Se applichi flip random qui, le coordinate RF-DETR non corrispondono più ai frame.

        if self.transform:

            if self.split == "train":

                if torch.rand(1).item() > 0.5:
                    frames = torch.stack([
                        F.hflip(f)
                        for f in frames
                    ])

                bright_factor = torch.empty(1).uniform_(0.6, 1.4).item()
                frames = torch.stack([
                    F.adjust_brightness(f, bright_factor)
                    for f in frames
                ])

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
            canestro = 0

        elif label_name in ["tiroDaDue1", "tiroDaTre1", "tiroLibero1"]:
            is_shot = True
            canestro = 1

        else:
            is_shot = False
            canestro = -1

        # ==========================
        # CONVERSIONE IN TENSORI
        # ==========================

        action_label = torch.tensor(action_label, dtype=torch.long)
        canestro = torch.tensor(canestro, dtype=torch.long)
        is_shot = torch.tensor(is_shot, dtype=torch.bool)

        return frames, mask, rfdetr_features, action_label, canestro, is_shot
