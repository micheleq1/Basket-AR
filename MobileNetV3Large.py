import torch
import torch.nn as nn
import torchvision.models as models
class MobileNetv3Large(nn.Module):
    def __init__(self):
        super().__init__()
        weights=models.MobileNet_V3_Large_Weights.DEFAULT
        mobilenet = models.mobilenet_v3_large(weights=weights)
        # V3 ha una struttura leggermente diversa
        self.features = mobilenet.features
        self.avgpool = mobilenet.avgpool
        # Il layer di proiezione interno di V3
        self.projection = nn.Sequential(
            mobilenet.classifier[0],  # Linear(960, 1280)
            mobilenet.classifier[1],  # Hardswish
        )

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        B, F, C, A, L = x.shape
        x = x.reshape(B * F, C, A, L)
        features = self.features(x)        # [B*F, 960, 7, 7]
        features = self.avgpool(features)  # [B*F, 960, 1, 1]
        features = torch.flatten(features, 1)   # [B*F, 960]
        features = self.projection(features)    # [B*F, 1280]
        return features.reshape(B, F, 1280)     # [B, F, 1280]
