import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import VideoDataset
import os
from torchvision import transforms
from MobileNetV2 import MobileNetv2
from GRUmodel import GRUmodel
import torch.nn as nn
import torch.optim as optim



def aggiorna_confusion_matrix(conf_matrix, labels, preds):
    """
    conf_matrix[classe_vera, classe_predetta]
    """
    labels = labels.cpu()
    preds = preds.cpu()

    for true_label, pred_label in zip(labels, preds):
        conf_matrix[true_label, pred_label] += 1

    return conf_matrix


def stampa_confusion_matrix(conf_matrix, idx_to_class):
    print("\nConfusion Matrix Validation")
    print("Righe = classe vera | Colonne = classe predetta\n")

    class_names = [
        idx_to_class[i]
        for i in range(len(idx_to_class))
    ]

    header = "vera\\pred".ljust(18)

    for name in class_names:
        header += name[:10].ljust(12)

    print(header)

    for i, row in enumerate(conf_matrix):
        line = class_names[i][:16].ljust(18)

        for value in row:
            line += str(value.item()).ljust(12)

        print(line)

    print()



FILE_ATTUALE = os.path.dirname(os.path.abspath(__file__))

DATASET_CARTELLA = os.path.abspath(
     os.path.join(FILE_ATTUALE, "..", "dataset")
)
MANIFEST = os.path.abspath(
     os.path.join(DATASET_CARTELLA, "manifest.csv")
 )

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

mobilenet_transforms = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])


train_dataset = VideoDataset(
    manifest_path=MANIFEST,
    video_dir=DATASET_CARTELLA,
    split="train",
    maxFrame=64,   
    imgSize=224,
    transform=mobilenet_transforms  
    )

validation_dataset = VideoDataset(
        manifest_path=MANIFEST,
        video_dir=DATASET_CARTELLA,
        split="val",
        maxFrame=64,   
        imgSize=224,
        transform=mobilenet_transforms  
    )   

test_dataset = VideoDataset(
        manifest_path=MANIFEST,
        video_dir=DATASET_CARTELLA,
        split="test",
        maxFrame=64,   
        imgSize=224,
        transform=mobilenet_transforms  
    )

    # prendiamo le label del train 

    
# Prendo le label testuali dei video nello split train
train_label_names = train_dataset.video_split.iloc[:, 5].values

# Converto ogni label testuale nel suo indice numerico
train_labels_numeric = np.array([
    train_dataset.class_to_idx[label_name]
    for label_name in train_label_names
], dtype=np.int64)

num_classes = len(train_dataset.class_to_idx)

# Conto quanti video ci sono per ogni classe nel train
class_count = np.bincount(
    train_labels_numeric,
    minlength=num_classes
)

print(f"-> Distribuzione classi nel Train: {class_count}")

# Calcolo i pesi inversi, gestendo eventuali classi con 0 elementi nel train
class_weights = np.zeros(num_classes, dtype=np.float32)

class_weights[class_count > 0] = 1.0 / class_count[class_count > 0]

print(f"-> Pesi classi: {class_weights}")

# Assegno a ogni video il peso della sua classe
sample_weights = class_weights[train_labels_numeric]

sample_weights = torch.DoubleTensor(sample_weights)
generator = torch.Generator()
generator.manual_seed(SEED)
# Sampler pesato
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    generator=generator,
    replacement=True
)
    
train_dataloader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
val_dataloader   = DataLoader(validation_dataset,   batch_size=16, shuffle=False)
test_dataloader  = DataLoader(test_dataset,  batch_size=16, shuffle=False)
conteggio_loader = torch.zeros(num_classes, dtype=torch.long)



"""
train_frames, train_masks, train_labels_out = next(iter(train_dataloader))
b_train, t_train, c_train, h_train, w_train = train_frames.shape
train_frames_per_mobilenet = train_frames.reshape(b_train * t_train, c_train, h_train, w_train)
    
val_frames, val_masks, val_labels_out = next(iter(val_dataloader))
b_val, t_val, c_val, h_val, w_val = val_frames.shape
val_frames_per_mobilenet = val_frames.reshape(b_val * t_val, c_val, h_val, w_val)

test_frames, test_masks, test_labels_out = next(iter(test_dataloader))
b_test, t_test, c_test, h_test, w_test = test_frames.shape
test_frames_per_mobilenet = test_frames.reshape(b_test * t_test, c_test, h_test, w_test)    

    #QUESTO NON L'HO CAPITO MA DAVA ERRORE E LO HA FATTO CHI NE SA 
    # Se train_labels_out è una tupla di stringhe/oggetti, la stampiamo semplicemente convertendola in lista
if isinstance(train_labels_out, tuple):
    print(f"Label estratte in questo batch: {list(train_labels_out)}")
else:
        # Se invece è un Tensor (numerico), usiamo il classico .tolist()
    print(f"Label estratte in questo batch: {train_labels_out.tolist()}")
        
print("Caro Michele, funziona, perché le classi rare tipo tiroDaTre vengono prese spesso")
"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device usato:", device)

mobilenet = MobileNetv2().to(device)
num_classes = len(train_dataset.class_to_idx)
grumodel = GRUmodel(
    input_size=1280,
    hidden_size=64,
    num_layers=1,
    num_classes=num_classes
).to(device)

# MobileNet è congelata: la usiamo solo per estrarre feature
mobilenet.eval()

# Loss per classificazione multiclasse
criterion = nn.CrossEntropyLoss()

# L'optimizer aggiorna SOLO la GRU, non MobileNet
optimizer = optim.AdamW(
    grumodel.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

num_epochs = 20
best_val_acc = 0.0

for epoch in range(num_epochs):

    # ==========================
    # TRAINING
    # ==========================

    grumodel.train()
    mobilenet.eval()

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for frames, masks, labels in train_dataloader:
        frames = frames.to(device)
        labels = labels.to(device).long()

        # Estraggo feature con MobileNet senza calcolare gradienti
        with torch.no_grad():
            features = mobilenet(frames)

        # features shape: [B, 32, 1280]
        logits = grumodel(features, masks)

        # logits shape: [B, num_classes]
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * frames.size(0)

        preds = torch.argmax(logits, dim=1)

        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_loss = train_loss_sum / train_total
    train_acc = train_correct / train_total

    # ==========================
    # VALIDATION
    # ==========================

    grumodel.eval()
    mobilenet.eval()

    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    val_conf_matrix = torch.zeros(
        num_classes,
        num_classes,
        dtype=torch.long
    )

    with torch.no_grad():
        for frames, masks, labels in val_dataloader:
            frames = frames.to(device)
            labels = labels.to(device).long()

            features = mobilenet(frames)

            logits = grumodel(features, masks)

            loss = criterion(logits, labels)

            val_loss_sum += loss.item() * frames.size(0)

            preds = torch.argmax(logits, dim=1)

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            val_conf_matrix = aggiorna_confusion_matrix(
                val_conf_matrix,
                labels,
                preds
        )

    val_loss = val_loss_sum / val_total
    val_acc = val_correct / val_total

    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
    print("-" * 50)
    stampa_confusion_matrix(
        val_conf_matrix,
        train_dataset.idx_to_class
)

    # Salvo il miglior modello
    if val_acc > best_val_acc:
        best_val_acc = val_acc

        torch.save({
            "gru_state_dict": grumodel.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "val_acc": val_acc,
            "val_loss": val_loss,
            "class_to_idx": train_dataset.class_to_idx,
            "idx_to_class": train_dataset.idx_to_class,
            "input_size": 1280,
            "hidden_size": 64,
            "num_layers": 1,
            "num_classes": num_classes
        }, "best_gru_basket_model.pth")

        print("Nuovo miglior modello salvato.")
