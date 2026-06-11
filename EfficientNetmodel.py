import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 usata SOLO come feature extractor.

    Input:
        x shape = [B, T, C, H, W]
        esempio = [batch, 32, 3, 704, 704]

    Output:
        features shape = [B, T, 1280]

    Nota:
        - Non fa fine-tuning.
        - Ridimensiona internamente ogni frame a 224x224.
        - Normalizza internamente con mean/std ImageNet.
    """

    def __init__(self):
        super().__init__()

        weights = models.EfficientNet_B0_Weights.DEFAULT
        efficientnet = models.efficientnet_b0(weights=weights)

        self.features = efficientnet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Congelo completamente EfficientNet: nessun fine tuning.
        for param in self.parameters():
            param.requires_grad = False

        # Mean e std ImageNet.
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        self.eval()

    def forward(self, x):
        """
        x deve essere in formato:
        [B, T, C, H, W]

        I frame possono essere 704x704.
        EfficientNet li riduce temporaneamente a 224x224.
        """

        B, T, C, H, W = x.shape

        # [B, T, C, H, W] -> [B*T, C, H, W]
        x = x.reshape(B * T, C, H, W)

        # Sicurezza: se arrivano valori 0-255 li porto a 0-1.
        x = x.float()
        if x.max() > 2.0:
            x = x / 255.0

        # Resize temporaneo per EfficientNet-B0.
        x = F.interpolate(
            x,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )
        x = (x - self.mean) / self.std
    

        # Feature extraction senza gradienti.
        with torch.no_grad():
            features = self.features(x)
            features = self.pool(features)
            features = torch.flatten(features, 1)

        # [B*T, 1280] -> [B, T, 1280]
        features = features.reshape(B, T, 1280)

        return features
