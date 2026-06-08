import torch
import torch.nn as nn
from transformers import AutoModel


class DINOv3FeatureExtractor(nn.Module):
    def __init__(
        self,
        model_name="facebook/dinov3-convnext-large-pretrain-lvd1689m",
        chunk_size=8
    ):
        super().__init__()

        self.model = AutoModel.from_pretrained(model_name)
        self.chunk_size = chunk_size

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        x shape: [B, T, C, H, W]
        x già normalizzato con mean/std DINOv3.
        """

        B, T, C, H, W = x.shape

        x = x.reshape(B * T, C, H, W)

        all_features = []

        for chunk in x.split(self.chunk_size, dim=0):
            outputs = self.model(pixel_values=chunk)

            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                features = outputs.pooler_output
            else:
                features = outputs.last_hidden_state.mean(dim=1)

            all_features.append(features)

        features = torch.cat(all_features, dim=0)

        features = features.reshape(B, T, -1)

        return features