import torch.nn as nn
import torchvision.models as models


class MobileNetv2(nn.modules):
    def __init__(self):
        super().__init__()
        



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

pesi=models.MobileNet_V2_Weights.DEFAULT
model=models.mobilenet_v2(pesi)

model=model.to(device)
model.eval()



