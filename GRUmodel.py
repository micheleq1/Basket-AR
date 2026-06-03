import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class GRUmodel(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers, num_classes ):
        """ input size: dimensione di ogni vettore, nel nostro caso di ogni frame, quindi 1280
            hidden_size: è la dimensione dello stato nascosto, ovvero di ogni output 
            num_layer: quanti strati di GRU vuoi impilare uno sopra l'altro

        """
        super().__init__()
        self.gru=nn.GRU(input_size,
                        hidden_size,
                        num_layers,
                        batch_first=True)
        self.dropout=nn.Dropout(0.3)
        self.fc1=nn.Linear(hidden_size, num_classes) # num_classes è il numero di classi da predire
        



    """dato che batch size=true, x deve essere nel formato [batch_size, sequence lenght, input size]
      dove sequence lenght è il numero di frame
    """
    def forward(self, x, mask):
            """
            x shape:    [batch_size, sequence_length, input_size]
                        esempio [16, 32, 1280]

            mask shape: [batch_size, sequence_length]
                        esempio [16, 32]
            """

            # Calcolo quanti frame reali ha ogni video
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

            packed_output, hidden = self.gru(packed_x)

            # hidden shape: [num_layers, batch_size, hidden_size]
            last_hidden = hidden[-1]

            last_hidden = self.dropout(last_hidden)

            logits = self.fc1(last_hidden)
            

            return logits