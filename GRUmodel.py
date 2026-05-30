
import torch.nn as nn

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
        self.fc1=nn.Linear(hidden_size, num_classes) # num_classes è il numero di classi da predire



    """dato che batch size=true, x deve essere nel formato [batch_size, sequence lenght, input size]
      dove sequence lenght è il numero di frame
    """
    def forward(self,x): 
        output,hidden=self.gru(x)
        last_hidden=hidden[-1]
        classi=self.fc1(last_hidden)
        return classi