import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from video_preprocessor import VideoPreprocessor
import torchvision.transforms.functional as F


class VideoDataset(Dataset):
    def __init__(self, manifest_path, video_dir, split, maxFrame, imgSize, transform=None):

        manifest = pd.read_csv(manifest_path)

        self.video_split = manifest[manifest["split"] == split].reset_index(drop=True)

        self.split = split
        self.video_dir = video_dir
        self.transform = transform
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
            "idle": "no_action",
            "non-gioco": "no_action",
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
        # PREPROCESSING VIDEO
        # ==========================

        frames, mask, total_frames = self.preprocessor(video_path)

        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

        mask = torch.from_numpy(mask).long()

        if self.transform:

            if self.split == "train" and label_name in self.classi_rare:

                # Flip orizzontale casuale
                if torch.rand(1).item() > 0.5:
                    frames = torch.stack([
                        F.hflip(f)
                        for f in frames
                    ])

                # Luminosità casuale
                bright_factor = torch.empty(1).uniform_(0.8, 1.2).item()

                frames = torch.stack([
                    F.adjust_brightness(f, bright_factor)
                    for f in frames
                ])

            # Normalizzazione MobileNet finale
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
        # CONVERSIONE IN TENSORI
        # ==========================

        action_label = torch.tensor(action_label, dtype=torch.long)

        # canestro deve essere long perché verrà usato con CrossEntropyLoss
        canestro = torch.tensor(canestro, dtype=torch.long)

        # is_shot deve essere bool perché serve come maschera
        is_shot = torch.tensor(is_shot, dtype=torch.bool)

        return frames, mask, action_label, canestro, is_shot