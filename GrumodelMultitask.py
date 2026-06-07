import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class GRUmodelMultitask(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_action_classes):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Aggiunto dropout=0.3 tra i layer della GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0.0
        )

        # Invece di una sola trasformazione lineare, separiamo i task con un mini-MLP
        # che aiuta a specializzare le feature per l'azione e per l'esito
        self.head_action = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, num_action_classes)
        )

        self.head_outcome = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, x, mask):
        lengths = mask.sum(dim=1)
        lengths = torch.clamp(lengths, min=1)
        lengths_cpu = lengths.cpu()

        packed_x = pack_padded_sequence(
            x,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, hidden = self.gru(packed_x)

        # Estraiamo gli ultimi stati forward e backward
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        last_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)

        # Predizioni separate dalle rispettive teste dedicate
        action_logits = self.head_action(last_hidden)
        outcome_logits = self.head_outcome(last_hidden)

        return action_logits, outcome_logits