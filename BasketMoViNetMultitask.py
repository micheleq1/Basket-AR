import torch
import torch.nn as nn

class BasketMoViNetMultitask(nn.Module):
    def __init__(self, num_action_classes, rfdetr_feature_dim=19):
        super().__init__()
        
        print("🚀 Caricamento di MoViNet-A0 (Video Mode) da Torch Hub...")
        # Carichiamo il backbone pre-addestrato su Kinetics-600 (azioni umane e sportive)
        self.backbone = torch.hub.load(
            'facebookresearch/pytorchvideo', 
            'movinet_a0_video', 
            pretrained=True
        )
        
        # MoViNet-A0 estrae 480 feature dal suo ultimo strato convoluzionale (conv_5)
        num_visual_features = self.backbone.head.conv_5.out_channels # 480
        
        # Sostituiamo il classificatore originale a 600 classi con un'identità
        # In questo modo il backbone ci sputerà direttamente il vettore di feature globali spatial-temporal pooled
        self.backbone.head.classifier = nn.Identity()
        
        # Dimensione totale del vettore fuso: Feature Visive (480) + Feature Tracking Roboflow (19)
        total_fused_dim = num_visual_features + rfdetr_feature_dim # 480 + 19 = 499
        
        # Sdoppiamento Multitask con MLP dedicati (ispirati alle vostre ultime modifiche sulla GRU)
        self.head_action = nn.Sequential(
            nn.Linear(total_fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5), # Alta regolarizzazione per salvare il Validation Score
            nn.Linear(256, num_action_classes)
        )
        
        self.head_outcome = nn.Sequential(
            nn.Linear(total_fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2) # 2 classi: Canestro sì (1) / Canestro no (0)
        )

    def forward(self, video, rfdetr_features):
        """
        Input attesi:
        - video: [Batch, Frame, Canali, Altezza, Larghezza] -> es. [16, 32, 3, 224, 224]
        - rfdetr_features: [Batch, Frame, 19] -> es. [16, 32, 19]
        """
        B, F, C, H, W = video.shape
        
        # ------------------------------------------------------------------
        # 1. ADATTAMENTO INPUT PER RETI VIDEO NATIVE
        # ------------------------------------------------------------------
        # Il vostro dataset produce i video come [B, F, C, H, W] (Dimensione Temporale prima dei Canali).
        # MoViNet vuole il formato nativo video 3D di PyTorchVideo: [Batch, Canali, Frame, Altezza, Larghezza]
        video_permuted = video.permute(0, 2, 1, 3, 4) # Diventa: [B, C, F, H, W] -> es. [16, 3, 32, 224, 224]
        
        # ------------------------------------------------------------------
        # 2. ESTRAZIONE FEATURE SPAZIO-TEMPORALI (BACKBONE)
        # ------------------------------------------------------------------
        # MoViNet analizza i 32 frame estraendo le dinamiche temporali e spaziali del tiro.
        # Output atteso dopo il Global Spatio-Temporal Average Pooling interno: [B, 480]
        visual_features = self.backbone(video_permuted) 
        
        # ------------------------------------------------------------------
        # 3. POOLING TEMPORALE DATI ROBOFLOW (RF-DETR)
        # ------------------------------------------------------------------
        # Le feature di Roboflow sono per-frame [B, 32, 19]. 
        # Applichiamo una media lungo la dimensione dei frame (dim=1) per ottenere un vettore 
        # riassuntivo della traiettoria globale di palla e canestro di dimensione [B, 19].
        rfdetr_pooled = torch.mean(rfdetr_features, dim=1)
        
        # ------------------------------------------------------------------
        # 4. FEATURE FUSION (CONCATENAZIONE)
        # ------------------------------------------------------------------
        # Uniamo la comprensione visiva del video con le coordinate geometriche pure del tracking
        fused_features = torch.cat([visual_features, rfdetr_pooled], dim=1) # Dimensione finale: [B, 499]
        
        # ------------------------------------------------------------------
        # 5. PREDIZIONE MULTITASK
        # ------------------------------------------------------------------
        out_action = self.head_action(fused_features)
        out_outcome = self.head_outcome(fused_features)
        
        return out_action, out_outcome