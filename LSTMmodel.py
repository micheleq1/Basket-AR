import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class LSTMmodel(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers, num_classes ):
        super().__init__()
        self.lstm=nn.LSTM(input_size,
                          hidden_size,
                          num_layers,
                          batch_first=True)
        self.dropout=nn.Dropout(0.3)
        self.fc1=nn.Linear(hidden_size,num_classes)

    def forward(self,x,mask):

            lengths = mask.sum(dim=1)

            # Sicurezza: evito lunghezze 0
            lengths = torch.clamp(lengths, min=1)

            # pack_padded_sequence vuole lengths su CPU
            lengths_cpu = lengths.cpu()

            # Impacchetto la sequenza ignorando il padding
            packed_x = pack_padded_sequence(
                x,
                lengths_cpu,
                batch_first=True,
                enforce_sorted=False
            )

            packed_output, hidden = self.lstm(packed_x)

            # hidden shape: [num_layers, batch_size, hidden_size]
            last_hidden = hidden[-1]

            last_hidden = self.dropout(last_hidden)

            logits = self.fc1(last_hidden)

            return logits