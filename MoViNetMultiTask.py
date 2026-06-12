import torch
import torch.nn as nn
# Assumendo l'uso del repository comune Atze00/MoViNet-pytorch
from movinets import MoViNet
from movinets.config import _C

class MoViNetRFDetrLiveFusion(nn.Module):
    def __init__(self, movinet_size=640, rfdetr_size=19, rfdetr_encoded_size=128, gru_hidden_size=256, num_action_classes=5):
        super().__init__()
        

        # 1. Carichiamo MoViNet per usarlo "dal vivo"
        self.video_backbone = MoViNet(_C.MODEL.MoViNetA2, causal=False, pretrained=True)
        self.video_backbone.classifier = nn.Identity()
        
        # SBLOCCHIAMO GLI ULTIMI STRATI (FINE-TUNING)
        # Congeliamo i primi blocchi (estraggono bordi, forme, parquet) per salvare memoria
        for param in self.video_backbone.parameters():
            param.requires_grad = False
            
        # Sblocchiamo solo l'ultimo blocco di MoViNet (il "b6") e l'head convoluzionale
        # Questo permette a MoViNet di imparare i pattern specifici del TUO campo e della TUA telecamera.
        for param in self.video_backbone.blocks[-1].parameters():
            param.requires_grad = True

        # 2. Modulo Geometrico (Invariato)
        self.rfdetr_encoder = nn.Sequential(
            nn.LayerNorm(rfdetr_size),
            nn.Linear(rfdetr_size, rfdetr_encoded_size),
            nn.ReLU(),
            nn.Dropout(0.10)
        )
        self.rfdetr_gru = nn.GRU(
            input_size=rfdetr_encoded_size,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        total_fusion_dim = movinet_size + (gru_hidden_size * 2)
        
        # 3. LayerNorm di Fusione (Che ha funzionato benissimo!)
        self.fusion_norm = nn.LayerNorm(total_fusion_dim)
        
        self.head_action = nn.Sequential(
            nn.Linear(total_fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(256, num_action_classes)
        )
        self.head_outcome = nn.Sequential(
            nn.Linear(total_fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 2)
        )

    def forward(self, video_frames, rfdetr_features, mask):
        # video_frames entra come [B, 32, 704, 704, 3] uint8
        
        # 1. Preparazione frame per MoViNet: [B, C, T, H, W]
        x = video_frames.permute(0, 4, 1, 2, 3).float() / 255.0
        
        # IMPORTANTE: MoViNet è stato addestrato a 224x224. Ridimensioniamo al volo.
        B, C, T, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = x.reshape(B, C, T, 224, 224)
        
        # ESTRAZIONE VIDEO "LIVE"
        movinet_feat = self.video_backbone(x) 

        # 2. Elaborazione ramo geometrico
        B_rf, T_rf, _ = rfdetr_features.shape
        rf_enc = self.rfdetr_encoder(rfdetr_features)
        lengths = mask.to(dtype=torch.bool).sum(dim=1).clamp(min=1, max=T_rf).cpu()
        packed_rf = pack_padded_sequence(rf_enc, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.rfdetr_gru(packed_rf)
        rf_geom_vector = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        # 3. FUSIONE RITARDATA E NORMALIZZATA
        fused_context = torch.cat([movinet_feat, rf_geom_vector], dim=-1)
        fused_context = self.fusion_norm(fused_context)
        
        # 4. OUTPUT
        action_logits = self.head_action(fused_context)
        outcome_logits = self.head_outcome(fused_context)
        
        return action_logits, outcome_logits
