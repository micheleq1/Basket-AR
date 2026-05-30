import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from video_preprocessor import VideoPreprocessor


class VideoDataset(Dataset):
    def __init__(self, manifest_path, video_dir, split, maxFrame, imgSize, transform=None):

        manifest = pd.read_csv(manifest_path)

        labels_complete = manifest.iloc[:, 5].values

        class_names = sorted(pd.unique(labels_complete))

        self.class_to_idx = {
            class_name: idx
            for idx, class_name in enumerate(class_names)
        }

        self.idx_to_class = {
            idx: class_name
            for class_name, idx in self.class_to_idx.items()
        }

        self.video_split = manifest[manifest["split"] == split].reset_index(drop=True)

        self.video_dir = video_dir
        self.transform = transform

        self.preprocessor = VideoPreprocessor(maxFrame, imgSize)

    def __len__(self):
        return len(self.video_split)

    def __getitem__(self, idx):
        rel_path = self.video_split.iloc[idx, 1]

        label_name = self.video_split.iloc[idx, 5]

        label = self.class_to_idx[label_name]

        video_path = os.path.normpath(
            os.path.join(self.video_dir, rel_path)
        )

        frames, mask, total_frames = self.preprocessor(video_path)

        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

        mask = torch.from_numpy(mask).long()

        if self.transform:
            frames = self.transform(frames)

        label = torch.tensor(label, dtype=torch.long)

        return frames, mask, label