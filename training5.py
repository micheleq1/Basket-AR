import os
from collections import Counter
from torchvision import transforms
import torch
import numpy as np
from torch.utils.data import DataLoader, Sampler
import torch.nn as nn
import torch.optim as optim

from dataset import VideoDataset
from EfficientNetmodel import EfficientNetB0
from GrumodelMultitask import GRUmodelMultitask


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


def mostra_classi_batch(dataloader, dataset, num_batch=3):
    print("\n==============================")
    print("CONTROLLO CLASSI NEI BATCH")
    print("==============================")

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batch:
            break

        frames, masks, rfdetr_features, action_labels, canestro, is_shot = batch

        action_names = [
            dataset.idx_to_action[int(label)]
            for label in action_labels
        ]

        conteggio_action = Counter(action_names)

        print(f"\nBatch {batch_idx + 1}")
        print("Distribuzione classi azione:")
        for classe, count in conteggio_action.items():
            print(f"  {classe}: {count}")

        print("Shape frames:", tuple(frames.shape))
        print("Shape RF-DETR features:", tuple(rfdetr_features.shape))

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

FILE_ATTUALE = os.path.dirname(os.path.abspath(__file__))

DATASET_CARTELLA = os.path.abspath(
    os.path.join(FILE_ATTUALE, "..", "dataset")
)

MANIFEST = os.path.abspath(
    os.path.join(DATASET_CARTELLA, "manifest.csv")
)

CACHE_FRAMES = os.path.abspath(
    os.path.join(DATASET_CARTELLA, "video_32_frame")
)

MASK_FRAMES = os.path.abspath(
    os.path.join(DATASET_CARTELLA, "mask_frame")
)

RFDETR_FEATURES = os.path.abspath(
    os.path.join(DATASET_CARTELLA, "rfdetr_features")
)

CHECKPOINT_PATH = os.path.join(
    FILE_ATTUALE,
    "best_multitask_basket_effnet_rfdetr.pth"
)


# ==========================
# CONFIG
# ==========================

SEED = 42
MAX_FRAMES = 32
IMG_SIZE = 704

BATCH_SIZE = 32
NUM_EPOCHS = 30

EFFICIENTNET_FEATURE_DIM = 1280
RFDETR_FEATURE_DIM = 19
GRU_INPUT_SIZE = EFFICIENTNET_FEATURE_DIM + RFDETR_FEATURE_DIM

HIDDEN_SIZE = 256
NUM_LAYERS = 2

LR = 0.001
WEIGHT_DECAY = 1e-4
LAMBDA_OUTCOME = 1.0


# ==========================
# SEED
# ==========================


# ==========================
# TRANSFORM MOBILENET
# ==========================

mobilenet_transforms = transforms.Compose([
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


# ==========================
# DATASET
# ==========================
# Nota importante:
# transform=None perché EfficientNet normalizza internamente.
# Inoltre non facciamo flip/brightness random qui, perché le feature RF-DETR
# sono già state estratte sui frame originali e devono rimanere allineate.

train_dataset = VideoDataset(
    manifest_path=MANIFEST,
    video_dir=DATASET_CARTELLA,
    cache_dir=CACHE_FRAMES,
    mask_dir=MASK_FRAMES,
    rfdetr_features_dir=RFDETR_FEATURES,
    rfdetr_feature_dim=RFDETR_FEATURE_DIM,
    split="train",
    maxFrame=MAX_FRAMES,
    imgSize=IMG_SIZE,
    transform=None
)

validation_dataset = VideoDataset(
    manifest_path=MANIFEST,
    video_dir=DATASET_CARTELLA,
    cache_dir=CACHE_FRAMES,
    mask_dir=MASK_FRAMES,
    rfdetr_features_dir=RFDETR_FEATURES,
    rfdetr_feature_dim=RFDETR_FEATURE_DIM,
    split="val",
    maxFrame=MAX_FRAMES,
    imgSize=IMG_SIZE,
    transform=None
)

test_dataset = VideoDataset(
    manifest_path=MANIFEST,
    video_dir=DATASET_CARTELLA,
    cache_dir=CACHE_FRAMES,
    mask_dir=MASK_FRAMES,
    rfdetr_features_dir=RFDETR_FEATURES,
    rfdetr_feature_dim=RFDETR_FEATURE_DIM,
    split="test",
    maxFrame=MAX_FRAMES,
    imgSize=IMG_SIZE,
    transform=None
)


# ==========================
# CLASSI MULTI-TASK
# ==========================

num_action_classes = len(train_dataset.action_to_idx)
num_outcome_classes = 2

print("Classi azione:", train_dataset.action_to_idx)
print("Classi esito:", train_dataset.outcome_to_idx)

# ==========================
# BALANCED SAMPLER SENZA REPLACEMENT
# ==========================

target_counts_per_epoch = {
    "idle":        400,
    "non-gioco":   500,
    "passaggio":   600,

    "tiroDaDue0":  197,
    "tiroDaDue1":  128,

    "tiroDaTre0":  111,
    "tiroDaTre1":   46,

    "tiroLibero0":  62,
    "tiroLibero1":  89,
}

print("\nTarget esempi per epoca:")
for cls, count in target_counts_per_epoch.items():
    print(f"  {cls}: {count}")



class BalancedNoReplacementSampler(Sampler):
    """
    Sampler bilanciato senza replacement.

    Ogni epoca:
    - prende al massimo target_counts_per_epoch[class_name] video per classe
    - non ripete lo stesso video dentro la stessa epoca
    - se una classe ha meno esempi del target, prende tutti gli esempi disponibili
    """

    def __init__(self, dataset, target_counts_per_epoch, seed=42):
        self.dataset = dataset
        self.target_counts_per_epoch = target_counts_per_epoch
        self.seed = seed
        self.epoch = 0

        self.labels = dataset.video_split.iloc[:, 5].values

        self.class_to_indices = {}

        for idx, label in enumerate(self.labels):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []

            self.class_to_indices[label].append(idx)

        for label in self.class_to_indices:
            self.class_to_indices[label] = np.array(
                self.class_to_indices[label],
                dtype=np.int64
            )

        self.length = 0

        for label, indices in self.class_to_indices.items():
            target = self.target_counts_per_epoch.get(label, len(indices))
            self.length += min(target, len(indices))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)

        selected_indices = []

        for label, indices in self.class_to_indices.items():
            target = self.target_counts_per_epoch.get(label, len(indices))

            n = min(target, len(indices))

            chosen = rng.choice(
                indices,
                size=n,
                replace=False
            )

            selected_indices.extend(chosen.tolist())

        rng.shuffle(selected_indices)

        self.epoch += 1

        return iter(selected_indices)

    def __len__(self):
        return self.length

sampler = BalancedNoReplacementSampler(
    dataset=train_dataset,
    target_counts_per_epoch=target_counts_per_epoch,
    seed=SEED
)
# ==========================
# DATALOADER
# ==========================

NUM_WORKERS = 8
PIN_MEMORY = True

dataloader_kwargs = {
    "num_workers": NUM_WORKERS,
    "pin_memory": PIN_MEMORY
}

if NUM_WORKERS > 0:
    dataloader_kwargs["persistent_workers"] = True
    dataloader_kwargs["prefetch_factor"] = 2

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    **dataloader_kwargs
)

val_dataloader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    **dataloader_kwargs
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    **dataloader_kwargs
)

mostra_classi_batch(
    train_dataloader,
    train_dataset,
    num_batch=3
)


# ==========================
# DEVICE
# ==========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device usato:", device)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    gpu_count = torch.cuda.device_count()
    print(f"GPU disponibili: {gpu_count}")

    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


# ==========================
# MODELLI
# ==========================

efficientnet = EfficientNetB0().to(device)

model = GRUmodelMultitask(
    input_size=GRU_INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_action_classes=num_action_classes
).to(device)

# Usa entrambe le GPU se disponibili
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print("Uso DataParallel su più GPU.")
    efficientnet = nn.DataParallel(efficientnet)
    model = nn.DataParallel(model)

efficientnet.eval()

print("Input GRU:", GRU_INPUT_SIZE)
print("  EfficientNet:", EFFICIENTNET_FEATURE_DIM)
print("  RF-DETR:", RFDETR_FEATURE_DIM)

# ==========================
# LOSS
# ==========================

criterion_action = nn.CrossEntropyLoss()
criterion_outcome = nn.CrossEntropyLoss()


# ==========================
# OPTIMIZER
# ==========================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)


# ==========================
# TRAINING
# ==========================

best_val_score = 0.0

for epoch in range(NUM_EPOCHS):

    # ==========================
    # TRAIN
    # ==========================

    model.train()
    efficientnet.eval()

    train_loss_sum = 0.0

    train_action_correct = 0
    train_action_total = 0

    train_outcome_correct = 0
    train_outcome_total = 0

    for frames, masks, rfdetr_features, action_labels, canestro, is_shot in train_dataloader:
        frames = frames.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        rfdetr_features = rfdetr_features.to(device, non_blocking=True).float()

        action_labels = action_labels.to(device, non_blocking=True).long()
        canestro = canestro.to(device, non_blocking=True).long()
        is_shot = is_shot.to(device, non_blocking=True)
        # EfficientNet è congelata: estrae solo feature visive.
        with torch.no_grad():
            visual_features = efficientnet(frames)

        # Concateno:
        # visual_features  = [B, 32, 1280]
        # rfdetr_features  = [B, 32, 19]
        # features         = [B, 32, 1299]
        features = torch.cat(
            [visual_features, rfdetr_features],
            dim=-1
        )

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

            loss = loss_action + LAMBDA_OUTCOME * loss_outcome
        else:
            loss = loss_action

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * frames.size(0)

        action_preds = torch.argmax(action_logits, dim=1)

        train_action_correct += (action_preds == action_labels).sum().item()
        train_action_total += action_labels.size(0)

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
    efficientnet.eval()

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
        for frames, masks, rfdetr_features, action_labels, canestro, is_shot in val_dataloader:
            frames = frames.to(device)
            masks = masks.to(device)
            rfdetr_features = rfdetr_features.to(device).float()

            action_labels = action_labels.to(device).long()
            canestro = canestro.to(device).long()
            is_shot = is_shot.to(device)

            visual_features = efficientnet(frames)

            features = torch.cat(
                [visual_features, rfdetr_features],
                dim=-1
            )

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

                loss = loss_action + LAMBDA_OUTCOME * loss_outcome
            else:
                loss = loss_action

            val_loss_sum += loss.item() * frames.size(0)

            action_preds = torch.argmax(action_logits, dim=1)

            val_action_correct += (action_preds == action_labels).sum().item()
            val_action_total += action_labels.size(0)

            val_action_conf_matrix = aggiorna_confusion_matrix(
                val_action_conf_matrix,
                action_labels,
                action_preds
            )

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

    val_score = 0.5 * val_action_acc + 0.5 * val_outcome_acc


    # ==========================
    # PRINT
    # ==========================

    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
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
    # SALVATAGGIO MIGLIOR MODELLO
    # ==========================

    if val_score > best_val_score:
        best_val_score = val_score

        model_to_save = model.module if isinstance(model, nn.DataParallel) else model

        torch.save({
            "model_type": "EfficientNetB0_RFDETR_GRU_MultiTask",
            "model_state_dict": model_to_save.state_dict(),
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

            "input_size": GRU_INPUT_SIZE,
            "efficientnet_feature_dim": EFFICIENTNET_FEATURE_DIM,
            "rfdetr_feature_dim": RFDETR_FEATURE_DIM,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "num_action_classes": num_action_classes,
            "maxFrame": MAX_FRAMES,
            "imgSize": IMG_SIZE,
            "backbone": "EfficientNetB0 frozen + RF-DETR features",
            "sampler": "BalancedNoReplacementSampler",
            "augmentation": "none_in_training_rfdetr_aligned"
        }, CHECKPOINT_PATH)