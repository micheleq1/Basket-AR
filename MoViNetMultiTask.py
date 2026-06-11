import torch
import torch.nn as nn
# Assumendo l'uso del repository comune Atze00/MoViNet-pytorch
from movinets import MoViNet
from movinets.config import _C

class MoViNetMultitask(nn.Module):
    def __init__(self, rfdetr_size=19, rfdetr_encoded_size=128, num_action_classes=5):
        super().__init__()
        
        # 1. Caricamento MoViNet-A2
        self.video_backbone = MoViNet(_C.MODEL.MoViNetA2, causal=False, pretrained=True)
        
        # Isoliamo la dimensione dell'output prima di rimuovere il classificatore di Kinetics
        num_features_video = self.video_backbone.classifier[0].in_features
        self.video_backbone.classifier = nn.Identity()
        
        # COGELAMENTO TOTALE: Nessun gradiente per MoViNet
        for param in self.video_backbone.parameters():
            param.requires_grad = False
            
        # 2. Modulo Geometrico per RF-DETR (Questo invece IMPARA)
        self.rfdetr_net = nn.Sequential(
            nn.LayerNorm(rfdetr_size),
            nn.Linear(rfdetr_size, rfdetr_encoded_size),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(rfdetr_encoded_size, rfdetr_encoded_size),
            nn.ReLU()
        )
        self.rfdetr_gru = nn.GRU(rfdetr_encoded_size, rfdetr_encoded_size, batch_first=True)

        # 3. Teste Multitask (Queste IMPARANO)
        total_features = num_features_video + rfdetr_encoded_size
        
        self.head_action = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(256, num_action_classes)
        )
        
        self.head_outcome = nn.Sequential(
            nn.Linear(total_features, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 2) # Canestro: Si (1) / No (0)
        )

    def forward(self, video_frames, rfdetr_features):
        # video_frames dal tuo dataset ha shape: [B, T, C, H, W]
        # MoViNet esige il canale colore PRIMA del tempo: [B, C, T, H, W]
        video_frames = video_frames.permute(0, 2, 1, 3, 4).float()
        
        if video_frames.max() > 2.0:
            video_frames = video_frames / 255.0
            
        # Estrazione feature video (senza calcolare gradienti per questa parte)
        with torch.no_grad():
            video_feat = self.video_backbone(video_frames) # Shape: [B, num_features_video]
        
        # Elaborazione Feature Geometriche
        rf_enc = self.rfdetr_net(rfdetr_features)
        _, h_n = self.rfdetr_gru(rf_enc)
        rf_feat = h_n[-1] # Ultimo stato nascosto della GRU geometrica
        
        # Fusione Finale (Late Fusion)
        x = torch.cat([video_feat, rf_feat], dim=-1)
        
        # Predizioni
        action_logits = self.head_action(x)
        outcome_logits = self.head_outcome(x)
        
        return action_logits, outcome_logits