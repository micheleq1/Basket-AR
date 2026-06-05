import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class GRUmodelMultitask(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_action_classes):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.3)

        # Bidirezionale: hidden_size * 2
        self.fc_action = nn.Linear(hidden_size * 2, num_action_classes)
        self.fc_outcome = nn.Linear(hidden_size * 2, 2)

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

        # hidden shape:
        # [num_layers * 2, batch_size, hidden_size]
        # Con bidirectional=True:
        # hidden[-2] = ultimo hidden forward dell'ultimo layer
        # hidden[-1] = ultimo hidden backward dell'ultimo layer

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        last_hidden = torch.cat(
            [forward_hidden, backward_hidden],
            dim=1
        )

        last_hidden = self.dropout(last_hidden)

        action_logits = self.fc_action(last_hidden)
        outcome_logits = self.fc_outcome(last_hidden)

        return action_logits, outcome_logits