import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetB0(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()

        weights = models.EfficientNet_B0_Weights.DEFAULT
        efficientnet = models.efficientnet_b0(weights=weights)

        self.features = efficientnet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x shape: [Batch, Frame, Canali, Altezza, Larghezza]
        B, F, C, H, W = x.shape

        # EfficientNet lavora su immagini, quindi trasformiamo i video in batch di frame
        x = x.reshape(B * F, C, H, W)

        features = self.features(x)              # [B*F, 1280, h, w]
        features = self.pool(features)           # [B*F, 1280, 1, 1]
        features = torch.flatten(features, 1)     # [B*F, 1280]

        features = features.reshape(B, F, 1280)  # [B, F, 1280]

        return features