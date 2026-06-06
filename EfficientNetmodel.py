import torch
import torch.nn as nn
import torchvision.models as models


# Nel EfficientNetmodel.py devi sbloccare gli ultimi blocchi
class EfficientNetB0(nn.Module):
    def __init__(self, finetune=True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT
        efficientnet = models.efficientnet_b0(weights=weights)
        self.features = efficientnet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Congela tutto
        for param in self.parameters():
            param.requires_grad = False

        # Sblocca ultimi 3 blocchi (indici 6, 7, 8)
        if finetune:
            for i, layer in enumerate(self.features):
                if i >= 6:
                    for param in layer.parameters():
                        param.requires_grad = True

    def forward(self, x):
        B, F, C, A, L = x.shape
        x = x.reshape(B * F, C, A, L)
        features = self.features(x)
        features = self.pool(features)
        features = torch.flatten(features, 1)
        return features.reshape(B, F, 1280)