import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from video_preprocessor import VideoPreprocessor

class VideoDataset(Dataset):
    def __init__(self, annotations_file, video_dir, split, maxFrame, imgSize, transform = None):

        all_labels = pd.read_csv(annotations_file)
        self.video_label = all_labels[all_labels['split'] == split].reset_index(drop=True)

        self.video_dir = video_dir
        self.transform = transform

        self.preprocessor = VideoPreprocessor(maxFrame, imgSize)

    def __len__(self):
        return len (self.video_label)

    def __getitem__(self, idx):
        rel_path = self.video_label.iloc[idx, 1]

        label = self.video_label.iloc[idx, 5]

        video_path = os.path.normpath(os.path.join(self.video_dir, rel_path))

        frames, mask, _= self.preprocessor(video_path);

        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        mask = torch.from_numpy(mask).long()

        if self.transform:
            frames = self.transform(frames)
            
        return frames, mask, label