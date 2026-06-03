import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class GRUmodelMultitask(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_action_classes):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)

        # Testa 1: tipo di azione
        self.fc_action = nn.Linear(hidden_size, num_action_classes)

        # Testa 2: esito tiro
        self.fc_outcome = nn.Linear(hidden_size, 2)

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

        last_hidden = hidden[-1]
        last_hidden = self.dropout(last_hidden)

        action_logits = self.fc_action(last_hidden)
        outcome_logits = self.fc_outcome(last_hidden)

        return action_logits, outcome_logits