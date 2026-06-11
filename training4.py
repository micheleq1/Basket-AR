import os
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

from dataset import VideoDataset
from MobileNetV2 import MobileNetv2
from GrumodelMultitask import GRUmodelMultitask
from collections import Counter
from MobileNetV3Large import MobileNetv3Large
from dinomodel import DINOv3FeatureExtractor
from transformers import AutoImageProcessor


# ==========================
# UTILITY
# ==========================

def aggiorna_confusion_matrix(conf_matrix, labels, preds):
    """
    conf_matrix[classe_vera, classe_predetta]
    """
    labels = labels.cpu()
    preds = preds.cpu()

    for true_label, pred_label in zip(labels, preds):
        conf_matrix[true_label, pred_label] += 1

    return conf_matrix


def stampa_confusion_matrix(conf_matrix, idx_to_class, titolo="Confusion Matrix Validation"):
    print(f"\n{titolo}")
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


def mostra_classi_batch(dataloader, dataset, num_batch=5):
    print("\n==============================")
    print("CONTROLLO CLASSI NEI BATCH")
    print("==============================")

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batch:
            break

        frames, masks, action_labels, canestro, is_shot = batch

        # Converto le action label numeriche in nomi leggibili
        action_names = [
            dataset.idx_to_action[int(label)]
            for label in action_labels
        ]

        conteggio_action = Counter(action_names)

        print(f"\nBatch {batch_idx + 1}")
        print("Distribuzione classi azione:")
        for classe, count in conteggio_action.items():
            print(f"  {classe}: {count}")

        # Controllo solo le clip che sono tiri
        shot_mask = is_shot.bool()

        print(f"Numero tiri nel batch: {shot_mask.sum().item()}")

        if shot_mask.sum() > 0:
            canestro_tiri = canestro[shot_mask]

            outcome_names = [
                dataset.idx_to_outcome[int(label)]
                for label in canestro_tiri
            ]

            conteggio_outcome = Counter(outcome_names)

            print("Distribuzione esito tiri:")
            for esito, count in conteggio_outcome.items():
                print(f"  {esito}: {count}")
        else:
            print("Nessun tiro in questo batch.")

# ==========================
# PATH
# ==========================

"""FILE_ATTUALE = os.path.dirname(os.path.abspath(__file__))

DATASET_CARTELLA = "/content/dataset_veloce/dataset"
MANIFEST = "/content/dataset_veloce/dataset/manifest.csv"

# Definiamo il percorso del checkpoint su Google Drive per non perderlo
CHECKPOINT_PATH = "/content/drive/MyDrive/ProgettoColab/best_multitask_basket_model.pth" """
FILE_ATTUALE = os.path.dirname(os.path.abspath(__file__))

DATASET_CARTELLA = os.path.abspath(
     os.path.join(FILE_ATTUALE, "..", "dataset")
)
MANIFEST = os.path.abspath(
     os.path.join(DATASET_CARTELLA, "manifest.csv")
 )

CHECKPOINT_PATH = os.path.join(
    FILE_ATTUALE,
    "best_multitask_basket.pth"
)
CACHE_FRAMES = os.path.abspath(os.path.join(DATASET_CARTELLA, "video_32_frame_imgsize_512"))
MASK_FRAMES = os.path.abspath(os.path.join(DATASET_CARTELLA, "mask_frame_imgsize_512"))

# ==========================
# SEED
# ==========================

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


# ==========================
# TRANSFORM MOBILENET
# ==========================

mobilenet_transforms = transforms.Compose([
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
model_name = "facebook/dinov3-convnext-large-pretrain-lvd1689m"

processor = AutoImageProcessor.from_pretrained(model_name)
dino_transform= transforms.Compose([
    transforms.Normalize(
        mean=processor.image_mean,
        std=processor.image_std
    )
])


# ==========================
# DATASET
# ==========================

train_dataset = VideoDataset(
    manifest_path=MANIFEST,
    video_dir=DATASET_CARTELLA,
    cache_dir=CACHE_FRAMES,
    mask_dir=MASK_FRAMES,
    split="train",
    maxFrame=32,
    imgSize=512,
    transform=dino_transform
)

validation_dataset = VideoDataset(
    manifest_path=MANIFEST,
    video_dir=DATASET_CARTELLA,
    cache_dir=CACHE_FRAMES,
    mask_dir=MASK_FRAMES,
    split="val",
    maxFrame=32,
    imgSize=512,
    transform=dino_transform
)

test_dataset = VideoDataset(
    manifest_path=MANIFEST,
    video_dir=DATASET_CARTELLA,
    cache_dir=CACHE_FRAMES,
    mask_dir=MASK_FRAMES,
    split="test",
    maxFrame=32,
    imgSize=512,
    transform=dino_transform
)


# ==========================
# CLASSI MULTI-TASK
# ==========================

num_action_classes = len(train_dataset.action_to_idx)
num_outcome_classes = 2

print("Classi azione:", train_dataset.action_to_idx)
print("Classi esito:", train_dataset.outcome_to_idx)


# ==========================
# WEIGHTED RANDOM SAMPLER
# ==========================

# Sostituisci tutta la sezione hardcodata con questo:
train_label_names = train_dataset.video_split.iloc[:, 5].values

# Calcola i conteggi reali dalle label del manifest
unique_labels, counts = np.unique(train_label_names, return_counts=True)
class_counts_train = dict(zip(unique_labels, counts))
print("Conteggi reali per classe:", class_counts_train)

# target_freq con le stesse chiavi esatte del manifest
target_freq = {
    'idle':        0.14,
    'non-gioco':   0.12,
    'passaggio':   0.32,   # ← aumenta
    'tiroDaDue0':  0.06,   # ← riduci
    'tiroDaDue1':  0.06,   # ← riduci
    'tiroDaTre0':  0.08,   # ← riduci leggermente
    'tiroDaTre1':  0.08,   # ← riduci leggermente
    'tiroLibero0': 0.08,   # ← riduci
    'tiroLibero1': 0.08,   # ← riduci
}

# Verifica che tutte le chiavi corrispondano
for label in unique_labels:
    if label not in target_freq:
        print(f"ATTENZIONE: label '{label}' non trovata in target_freq!")

class_weights = {
    cls: target_freq[cls] / class_counts_train[cls]
    for cls in class_counts_train
}

sample_weights = [class_weights[label] for label in train_label_names]
print("Pesi calcolati:")
for cls, w in class_weights.items():
    print(f"  {cls}: {w:.6f}")
sample_weights = torch.DoubleTensor(sample_weights)

generator = torch.Generator()
generator.manual_seed(SEED)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
    generator=generator
)


# ==========================
# DATALOADER
# ==========================

train_dataloader = DataLoader(
    train_dataset,
    batch_size=8,
    sampler=sampler
)

val_dataloader = DataLoader(
    validation_dataset,
    batch_size=8,
    shuffle=False
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)
mostra_classi_batch(
    train_dataloader,
    train_dataset,
    num_batch=5
)

# ==========================
# DEVICE
# ==========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device usato:", device)


# ==========================
# MODELLI
# ==========================


dinov3= DINOv3FeatureExtractor().to(device)

frames_test, masks_test, action_labels_test, canestro_test, is_shot_test = next(iter(train_dataloader))
frames_test = frames_test[:1].to(device)  # solo 1 video per non saturare la GPU

with torch.no_grad():
    test_features = dinov3(frames_test)

dino_feature_dim = test_features.shape[-1]

print("Shape feature DINOv3:", test_features.shape)
print("Dimensione feature DINOv3:", dino_feature_dim)

model = GRUmodelMultitask(
    input_size=dino_feature_dim,
    hidden_size=256,
    num_layers=2,
    num_action_classes=num_action_classes
).to(device)


# --- LOGICA DI CARICAMENTO DEI PESI COMPATIBILE CON IL RESUME ---
start_epoch = 0
best_val_score = 0.0
checkpoint = None

if os.path.exists(CHECKPOINT_PATH):
    print(f"-> Trovato checkpoint in {CHECKPOINT_PATH}. Caricamento in corso...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    start_epoch = checkpoint["epoch"]
    best_val_score = checkpoint.get("val_score", 0.0)
    print(f"-> Ripristino completato! Si riparte dall'epoca {start_epoch + 1} con Val Score di riferimento: {best_val_score:.4f}")
else:
    print("-> Nessun checkpoint trovato. L'addestramento partirà dall'inizio.")
# ----------------------------------------------------------------


# ==========================
# LOSS
# ==========================

criterion_action = nn.CrossEntropyLoss(

)

criterion_outcome = nn.CrossEntropyLoss(
    
)

# Peso della loss dell'esito del tiro.
lambda_outcome = 1.0


# ==========================
# OPTIMIZER
# ==========================

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

# Se è stato caricato un checkpoint, ripristiniamo anche lo stato dell'ottimizzatore
if checkpoint is not None and "optimizer_state_dict" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


# ==========================
# TRAINING
# ==========================

num_epochs = 20

# Il ciclo ora parte da start_epoch (0 se nuovo, o l'indice salvato se ripristinato)
for epoch in range(start_epoch, num_epochs):

    # ==========================
    # TRAINING
    # ==========================

    model.train()
    dinov3.eval()

    train_loss_sum = 0.0

    train_action_correct = 0
    train_action_total = 0

    train_outcome_correct = 0
    train_outcome_total = 0

    for frames, masks, action_labels, canestro, is_shot in train_dataloader:
        frames = frames.to(device)
        masks = masks.to(device)
        action_labels = action_labels.to(device).long()
        canestro = canestro.to(device).long()
        is_shot = is_shot.to(device)

        with torch.no_grad():
            features = dinov3(frames)

        action_logits, outcome_logits = model(features, masks)

        # Loss azione: calcolata su tutte le clip.
        loss_action = criterion_action(
            action_logits,
            action_labels
        )

        # Loss esito: calcolata solo sulle clip che sono tiri.
        shot_mask = is_shot == True

        if shot_mask.sum() > 0:
            loss_outcome = criterion_outcome(
                outcome_logits[shot_mask],
                canestro[shot_mask]
            )

            loss = loss_action + lambda_outcome * loss_outcome
        else:
            loss = loss_action

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * frames.size(0)

        # Accuracy azione
        action_preds = torch.argmax(action_logits, dim=1)

        train_action_correct += (action_preds == action_labels).sum().item()
        train_action_total += action_labels.size(0)

        # Accuracy esito solo sui tiri
        if shot_mask.sum() > 0:
            outcome_preds = torch.argmax(
                outcome_logits[shot_mask],
                dim=1
            )

            train_outcome_correct += (
                outcome_preds == canestro[shot_mask]
            ).sum().item()

            train_outcome_total += shot_mask.sum().item()

    train_loss = train_loss_sum / train_action_total
    train_action_acc = train_action_correct / train_action_total

    if train_outcome_total > 0:
        train_outcome_acc = train_outcome_correct / train_outcome_total
    else:
        train_outcome_acc = 0.0


    # ==========================
    # VALIDATION
    # ==========================

    model.eval()
    dinov3.eval()

    val_loss_sum = 0.0

    val_action_correct = 0
    val_action_total = 0

    val_outcome_correct = 0
    val_outcome_total = 0

    val_action_conf_matrix = torch.zeros(
        num_action_classes,
        num_action_classes,
        dtype=torch.long
    )

    val_outcome_conf_matrix = torch.zeros(
        num_outcome_classes,
        num_outcome_classes,
        dtype=torch.long
    )

    with torch.no_grad():
        for frames, masks, action_labels, canestro, is_shot in val_dataloader:
            frames = frames.to(device)
            masks = masks.to(device)
            action_labels = action_labels.to(device).long()
            canestro = canestro.to(device).long()
            is_shot = is_shot.to(device)

            features = dinov3(frames)

            action_logits, outcome_logits = model(features, masks)

            loss_action = criterion_action(
                action_logits,
                action_labels
            )

            shot_mask = is_shot == True

            if shot_mask.sum() > 0:
                loss_outcome = criterion_outcome(
                    outcome_logits[shot_mask],
                    canestro[shot_mask]
                )

                loss = loss_action + lambda_outcome * loss_outcome
            else:
                loss = loss_action

            val_loss_sum += loss.item() * frames.size(0)

            # Accuracy azione
            action_preds = torch.argmax(action_logits, dim=1)

            val_action_correct += (action_preds == action_labels).sum().item()
            val_action_total += action_labels.size(0)

            val_action_conf_matrix = aggiorna_confusion_matrix(
                val_action_conf_matrix,
                action_labels,
                action_preds
            )

            # Accuracy esito solo sui tiri
            if shot_mask.sum() > 0:
                outcome_preds = torch.argmax(
                    outcome_logits[shot_mask],
                    dim=1
                )

                val_outcome_correct += (
                    outcome_preds == canestro[shot_mask]
                ).sum().item()

                val_outcome_total += shot_mask.sum().item()

                val_outcome_conf_matrix = aggiorna_confusion_matrix(
                    val_outcome_conf_matrix,
                    canestro[shot_mask],
                    outcome_preds
                )

    val_loss = val_loss_sum / val_action_total
    val_action_acc = val_action_correct / val_action_total

    if val_outcome_total > 0:
        val_outcome_acc = val_outcome_correct / val_outcome_total
    else:
        val_outcome_acc = 0.0

    # Score complessivo per salvare il miglior modello.
    val_score = 0.5 * val_action_acc + 0.5 * val_outcome_acc


    # ==========================
    # PRINT RISULTATI
    # ==========================

    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Action Acc:  {train_action_acc:.4f}")
    print(f"Train Outcome Acc: {train_outcome_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Action Acc:  {val_action_acc:.4f}")
    print(f"Val Outcome Acc: {val_outcome_acc:.4f}")
    print(f"Val Score: {val_score:.4f}")
    print("-" * 50)

    stampa_confusion_matrix(
        val_action_conf_matrix,
        train_dataset.idx_to_action,
        titolo="Confusion Matrix Validation - Azione"
    )

    stampa_confusion_matrix(
        val_outcome_conf_matrix,
        train_dataset.idx_to_outcome,
        titolo="Confusion Matrix Validation - Esito Tiro"
    )


    # ==========================
    # SALVATAGGIO MIGLIOR MODELLO (Modificato per salvare su Google Drive)
    # ==========================

    if val_score > best_val_score:
        best_val_score = val_score

        torch.save({
            "model_type": "MobileNetV2_GRU_MultiTask",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,

            "val_score": val_score,
            "val_loss": val_loss,
            "val_action_acc": val_action_acc,
            "val_outcome_acc": val_outcome_acc,

            "action_to_idx": train_dataset.action_to_idx,
            "idx_to_action": train_dataset.idx_to_action,
            "outcome_to_idx": train_dataset.outcome_to_idx,
            "idx_to_outcome": train_dataset.idx_to_outcome,

            "input_size": 1280,
            "hidden_size": 64,
            "num_layers": 1,
            "num_action_classes": num_action_classes,
            "maxFrame": 48,
            "backbone": "MobileNetV2",
            "sampler": "WeightedRandomSampler",
            "augmentation": "rare_classes_flip_brightness"
        }, CHECKPOINT_PATH)  

        print("Nuovo miglior modello multi-task salvato su Google Drive.")