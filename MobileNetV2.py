import torch.nn as nn
import torchvision.models as models
import torch

class MobileNetv2(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT
        mobilenet = models.mobilenet_v2(weights=weights)
        self.features = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        for param in self.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        # x inizialmente ha dimensione: [Batch_size, Frame, Canali, Altezza, Larghezza]
        B,F,C,A,L=x.shape
        # cosi ogni immagine viene processata in modo indipendente, perche mobilenet non puo processare video
        x=x.reshape(B*F,C,A,L)

        features = self.features(x)  # Output: [Batch*Frame, 1280, 7, 7]
        features = self.pool(features)      # Output: [Batch*Frame, 1280, 1, 1]
        features = torch.flatten(features, 1) # Output: [Batch*Frame, 1280]
        features=features.reshape(B,F,1280) # Output: [Batch, Frame, 1280] quindi viene ricostruita la struttura video-frame, dove ogni frame è descritto da un vettore di 1280 valori


        return features


