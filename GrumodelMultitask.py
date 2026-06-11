import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class GRUmodelMultitask(nn.Module):
    def __init__(
        self,
        visual_size: int,
        rfdetr_size: int,
        hidden_size: int,
        num_layers: int,
        num_action_classes: int,
        rfdetr_encoded_size: int = 128,
        gru_dropout: float = 0.15,
        head_dropout: float = 0.20,
    ):
        super().__init__()

        self.visual_size = visual_size
        self.rfdetr_size = rfdetr_size
        self.rfdetr_encoded_size = rfdetr_encoded_size

        # Le feature RF-DETR (19 valori) vengono portate a 128 dimensioni.
        self.rfdetr_encoder = nn.Sequential(
            nn.LayerNorm(rfdetr_size),
            nn.Linear(rfdetr_size, rfdetr_encoded_size),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(rfdetr_encoded_size, rfdetr_encoded_size),
            nn.ReLU(),
        )

        # Stabilizza le feature visive prima della fusione.
        self.visual_norm = nn.LayerNorm(visual_size)

        self.gru = nn.GRU(
            input_size=visual_size + rfdetr_encoded_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout if num_layers > 1 else 0.0,
        )

        self.dropout_action = nn.Dropout(head_dropout)
        self.dropout_outcome = nn.Dropout(head_dropout)

        self.fc_action = nn.Linear(hidden_size * 2, num_action_classes)
        self.fc_outcome = nn.Linear(hidden_size * 2, 2)

    def forward(
        self,
        visual_features: torch.Tensor,
        rfdetr_features: torch.Tensor,
        mask: torch.Tensor,
    ):
        """
        visual_features: [B, T, visual_size]
        rfdetr_features: [B, T, rfdetr_size]
        mask:            [B, T], con 1 per frame reale e 0 per padding
        """
        if visual_features.ndim != 3:
            raise ValueError(
                f"visual_features deve avere shape [B,T,F], ricevuto {visual_features.shape}"
            )
        if rfdetr_features.ndim != 3:
            raise ValueError(
                f"rfdetr_features deve avere shape [B,T,F], ricevuto {rfdetr_features.shape}"
            )
        if mask.ndim != 2:
            raise ValueError(f"mask deve avere shape [B,T], ricevuto {mask.shape}")

        if visual_features.shape[:2] != rfdetr_features.shape[:2]:
            raise ValueError(
                "Feature visive e RF-DETR non allineate: "
                f"{visual_features.shape} vs {rfdetr_features.shape}"
            )
        if visual_features.shape[:2] != mask.shape:
            raise ValueError(
                f"Feature e mask non allineate: {visual_features.shape} vs {mask.shape}"
            )

        batch_size, timesteps, _ = rfdetr_features.shape

        visual_features = self.visual_norm(visual_features)

        rfdetr_flat = rfdetr_features.reshape(batch_size * timesteps, -1)
        rfdetr_encoded = self.rfdetr_encoder(rfdetr_flat)
        rfdetr_encoded = rfdetr_encoded.reshape(
            batch_size, timesteps, self.rfdetr_encoded_size
        )

        x = torch.cat([visual_features, rfdetr_encoded], dim=-1)

        mask = mask.to(dtype=torch.bool)
        lengths = mask.sum(dim=1)
        lengths = torch.clamp(lengths, min=1, max=timesteps)
        lengths_cpu = lengths.to(device="cpu", dtype=torch.long)

        packed_x = pack_padded_sequence(
            x,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed_x)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        last_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)

        action_logits = self.fc_action(self.dropout_action(last_hidden))
        outcome_logits = self.fc_outcome(self.dropout_outcome(last_hidden))

        return action_logits, outcome_logits
