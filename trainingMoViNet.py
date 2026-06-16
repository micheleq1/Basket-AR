import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

from dataset import VideoDataset
from movinets import MoViNet
from movinets.config import _C

# ============================================================
# NUOVO MODELLO: LIVE FINE-TUNING (MoViNet + RF-DETR)
# ============================================================

class MoViNetRFDetrLiveFusion(nn.Module):
    def __init__(self, rfdetr_size=19, rfdetr_encoded_size=128, gru_hidden_size=256, num_action_classes=5):
        super().__init__()
        
        # 1. Carichiamo MoViNet per usarlo "dal vivo"
        self.video_backbone = MoViNet(_C.MODEL.MoViNetA2, causal=False, pretrained=True)
        movinet_out_features = self.video_backbone.classifier[0].in_features
        self.video_backbone.classifier = nn.Identity()
        
        # SBLOCCHIAMO GLI ULTIMI STRATI (FINE-TUNING)
        # Congeliamo i primi blocchi (estraggono bordi, forme, parquet) per salvare memoria
        for param in self.video_backbone.parameters():
            param.requires_grad = False
            
        # Sblocchiamo solo l'ultimo blocco di MoViNet (il "b6") e l'head convoluzionale
        # Questo permette a MoViNet di imparare i pattern specifici del TUO campo e della TUA telecamera.
        for param in self.video_backbone.blocks[-1].parameters():
            param.requires_grad = True
        for param in self.video_backbone.conv_head.parameters():
            param.requires_grad = True

        # 2. Modulo Geometrico (Invariato)
        self.rfdetr_encoder = nn.Sequential(
            nn.LayerNorm(rfdetr_size),
            nn.Linear(rfdetr_size, rfdetr_encoded_size),
            nn.ReLU(),
            nn.Dropout(0.10)
        )
        self.rfdetr_gru = nn.GRU(
            input_size=rfdetr_encoded_size,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        total_fusion_dim = movinet_out_features + (gru_hidden_size * 2)
        
        # 3. LayerNorm di Fusione (Che ha funzionato benissimo!)
        self.fusion_norm = nn.LayerNorm(total_fusion_dim)
        
        self.head_action = nn.Sequential(
            nn.Linear(total_fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(256, num_action_classes)
        )
        self.head_outcome = nn.Sequential(
            nn.Linear(total_fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 2)
        )

    def forward(self, video_frames, rfdetr_features, mask):
        # video_frames entra come [B, 32, 704, 704, 3] uint8
        
        # 1. Preparazione frame per MoViNet: [B, C, T, H, W]
        x = video_frames.permute(0, 4, 1, 2, 3).float() / 255.0
        
        # IMPORTANTE: MoViNet è stato addestrato a 224x224. Ridimensioniamo al volo.
        B, C, T, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = x.reshape(B, C, T, 224, 224)
        
        # ESTRAZIONE VIDEO "LIVE"
        movinet_feat = self.video_backbone(x) 

        # 2. Elaborazione ramo geometrico
        B_rf, T_rf, _ = rfdetr_features.shape
        rf_enc = self.rfdetr_encoder(rfdetr_features)
        lengths = mask.to(dtype=torch.bool).sum(dim=1).clamp(min=1, max=T_rf).cpu()
        packed_rf = pack_padded_sequence(rf_enc, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.rfdetr_gru(packed_rf)
        rf_geom_vector = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        # 3. FUSIONE RITARDATA E NORMALIZZATA
        fused_context = torch.cat([movinet_feat, rf_geom_vector], dim=-1)
        fused_context = self.fusion_norm(fused_context)
        
        # 4. OUTPUT
        action_logits = self.head_action(fused_context)
        outcome_logits = self.head_outcome(fused_context)
        
        return action_logits, outcome_logits


# ============================================================
# CONFIGURAZIONE
# ============================================================

FILE_ATTUALE = os.path.dirname(os.path.abspath(__file__))

DATASET_CARTELLA = os.path.abspath(
    os.path.join(FILE_ATTUALE, "..", "dataset")
)

MANIFEST = os.path.join(DATASET_CARTELLA, "manifest.csv")
CACHE_FRAMES = os.path.join(DATASET_CARTELLA, "video_32_frame")
MASK_FRAMES = os.path.join(DATASET_CARTELLA, "mask_frame")
RFDETR_FEATURES = os.path.join(DATASET_CARTELLA, "rfdetr_features")

CHECKPOINT_PATH = os.path.join(
    FILE_ATTUALE,
    "best_multitask_basket_movinet_live.pth",
)

SEED = 42

MAX_FRAMES = 32
IMG_SIZE = 704

# BATCH_SIZE abbassato a 2 per evitare Out Of Memory con il video in 3D
BATCH_SIZE = 2
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 8

NUM_WORKERS = 4
PIN_MEMORY = True

MOVINET_FEATURE_DIM = 640
RFDETR_FEATURE_DIM = 19
RFDETR_ENCODED_DIM = 128

HIDDEN_SIZE = 256

LR = 1e-4 # Learning rate leggermente ridotto per il fine-tuning
WEIGHT_DECAY = 1e-4
LAMBDA_OUTCOME = 1.0
GRAD_CLIP_NORM = 1.0

USE_AMP = True


# ============================================================
# RIPRODUCIBILITÀ
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True


# ============================================================
# UTILITY METRICHE
# ============================================================

def aggiorna_confusion_matrix(conf_matrix: torch.Tensor, labels: torch.Tensor, preds: torch.Tensor) -> None:
    labels_cpu = labels.detach().cpu()
    preds_cpu = preds.detach().cpu()
    for true_label, pred_label in zip(labels_cpu, preds_cpu):
        conf_matrix[int(true_label), int(pred_label)] += 1

def recall_per_classe(conf_matrix: torch.Tensor) -> torch.Tensor:
    support = conf_matrix.sum(dim=1).float()
    true_positive = torch.diag(conf_matrix).float()
    recall = torch.zeros_like(support)
    valid = support > 0
    recall[valid] = true_positive[valid] / support[valid]
    return recall

def stampa_metriche_classi(conf_matrix: torch.Tensor, idx_to_class: dict, titolo: str) -> None:
    print(f"\n{titolo}")
    print("Righe = classe vera | Colonne = classe predetta\n")
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    # Questo approccio evita l'errore del backslash '\' nelle f-string!
    header = "vera\\pred".ljust(18)
    for name in class_names:
        header += name[:10].ljust(12)
    print(header)
    
    for i, row in enumerate(conf_matrix):
        line = class_names[i][:16].ljust(18)
        for value in row:
            line += str(int(value.item())).ljust(12)
        print(line)
        
    recalls = recall_per_classe(conf_matrix)
    print("\nRecall per classe:")
    for i, value in enumerate(recalls):
        print(f"  {class_names[i]}: {value.item():.4f}")
    valid = conf_matrix.sum(dim=1) > 0
    macro_recall = recalls[valid].mean().item() if valid.any() else 0.0
    print(f"Macro recall: {macro_recall:.4f}")


# ============================================================
# CONTROLLI FILE E DATASET
# ============================================================

def controlla_percorsi() -> None:
    richiesti = [MANIFEST, CACHE_FRAMES, MASK_FRAMES, RFDETR_FEATURES]
    for percorso in richiesti:
        if not os.path.exists(percorso):
            raise FileNotFoundError(f"Percorso mancante: {percorso}")

def nome_cache_da_rel_path(rel_path: str) -> str:
    return rel_path.replace("/", "_").replace("\\", "_") + ".npy"

def controlla_feature_rfdetr(dataset: VideoDataset, split: str) -> None:
    mancanti = []
    for rel_path in dataset.video_split.iloc[:, 1].values:
        nome_file = nome_cache_da_rel_path(rel_path)
        path_feature = os.path.join(RFDETR_FEATURES, nome_file)
        if not os.path.exists(path_feature):
            mancanti.append(path_feature)
    if mancanti:
        esempi = "\n".join(f"  {x}" for x in mancanti[:10])
        raise FileNotFoundError(
            f"Mancano {len(mancanti)} file RF-DETR nello split '{split}'.\n"
            f"Primi esempi:\n{esempi}"
        )
    print(f"Feature RF-DETR complete per lo split '{split}'.")

controlla_percorsi()

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
    transform=None,
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
    transform=None,
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
    transform=None,
)

controlla_feature_rfdetr(train_dataset, "train")
controlla_feature_rfdetr(validation_dataset, "val")
controlla_feature_rfdetr(test_dataset, "test")

num_action_classes = len(train_dataset.action_to_idx)
num_outcome_classes = 2

print("\nClassi azione:", train_dataset.action_to_idx)
print("Classi esito:", train_dataset.outcome_to_idx)


# ============================================================
# DATALOADER
# ============================================================

dataloader_kwargs = {
    "num_workers": NUM_WORKERS,
    "pin_memory": PIN_MEMORY,
}

if NUM_WORKERS > 0:
    dataloader_kwargs["persistent_workers"] = True
    dataloader_kwargs["prefetch_factor"] = 1

train_generator = torch.Generator()
train_generator.manual_seed(SEED)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=train_generator,
    **dataloader_kwargs,
)

val_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, **dataloader_kwargs)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **dataloader_kwargs)


# ============================================================
# DEVICE E GPU
# ============================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
amp_enabled = USE_AMP and device.type == "cuda"

print("\nDevice principale:", device)

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"GPU disponibili: {gpu_count}")
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")


# ============================================================
# PESI DELLE LOSS
# ============================================================

action_counts = np.zeros(num_action_classes, dtype=np.float32)
for original_label in train_dataset.video_split.iloc[:, 5].values:
    action_name = train_dataset.action_mapping[original_label]
    action_idx = train_dataset.action_to_idx[action_name]
    action_counts[action_idx] += 1

if np.any(action_counts == 0):
    raise RuntimeError(f"Una classe azione non ha esempi nel train: {action_counts}")

action_weights_np = 1.0 / np.sqrt(action_counts)
action_weights_np = action_weights_np / action_weights_np.mean()
action_weights = torch.tensor(action_weights_np, dtype=torch.float32, device=device)

outcome_counts = np.zeros(num_outcome_classes, dtype=np.float32)
for original_label in train_dataset.video_split.iloc[:, 5].values:
    if original_label in {"tiroDaDue0", "tiroDaTre0", "tiroLibero0"}:
        outcome_counts[0] += 1
    elif original_label in {"tiroDaDue1", "tiroDaTre1", "tiroLibero1"}:
        outcome_counts[1] += 1

if np.any(outcome_counts == 0):
    raise RuntimeError(f"Una classe esito non ha esempi nel train: {outcome_counts}")

outcome_weights_np = 1.0 / np.sqrt(outcome_counts)
outcome_weights_np = outcome_weights_np / outcome_weights_np.mean()
outcome_weights = torch.tensor(outcome_weights_np, dtype=torch.float32, device=device)


# ============================================================
# MODELLI
# ============================================================

# Ora istanziamo la versione "Live"
model = MoViNetRFDetrLiveFusion(
    rfdetr_size=RFDETR_FEATURE_DIM,
    rfdetr_encoded_size=RFDETR_ENCODED_DIM,
    gru_hidden_size=HIDDEN_SIZE,
    num_action_classes=num_action_classes
).to(device)

print(
    "\nDimensioni architettura: "
    f"MoViNet={MOVINET_FEATURE_DIM}, "
    f"RF-DETR raw={RFDETR_FEATURE_DIM}, "
    f"RF-DETR encoded={RFDETR_ENCODED_DIM}, "
    f"GRU Hidden={HIDDEN_SIZE}"
)


# ============================================================
# LOSS, OPTIMIZER, AMP, SCHEDULER
# ============================================================

criterion_action = nn.CrossEntropyLoss(weight=action_weights)
criterion_outcome = nn.CrossEntropyLoss(weight=outcome_weights)

# IMPORTANTE: Ottimizziamo SOLO i parametri sbloccati (risparmia tantissima VRAM!)
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=LR, 
    weight_decay=WEIGHT_DECAY
)
scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
)


# ============================================================
# ESECUZIONE DI UN'EPOCA
# ============================================================

def esegui_epoca(dataloader: DataLoader, training: bool):
    if training:
        model.train()
    else:
        model.eval()

    loss_sum = 0.0
    action_correct = 0
    action_total = 0
    outcome_correct = 0
    outcome_total = 0

    action_conf_matrix = torch.zeros(num_action_classes, num_action_classes, dtype=torch.long)
    outcome_conf_matrix = torch.zeros(num_outcome_classes, num_outcome_classes, dtype=torch.long)

    grad_context = torch.enable_grad() if training else torch.no_grad()

    with grad_context:
        for (
            frames, # RICEVIAMO I FRAME ORIGINALI DAL DATASET
            masks,
            rfdetr_features,
            action_labels,
            canestro,
            is_shot,
        ) in dataloader:

            frames = frames.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            rfdetr_features = rfdetr_features.to(device, non_blocking=True, dtype=torch.float32)
            action_labels = action_labels.to(device, non_blocking=True, dtype=torch.long)
            canestro = canestro.to(device, non_blocking=True, dtype=torch.long)
            is_shot = is_shot.to(device, non_blocking=True, dtype=torch.bool)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                # Passiamo i FRAME VERI al modello
                action_logits, outcome_logits = model(
                    frames,
                    rfdetr_features,
                    masks,
                )

                loss_action = criterion_action(action_logits, action_labels)
                shot_mask = is_shot

                if shot_mask.any():
                    loss_outcome = criterion_outcome(
                        outcome_logits[shot_mask],
                        canestro[shot_mask],
                    )
                    loss = loss_action + LAMBDA_OUTCOME * loss_outcome
                else:
                    loss = loss_action

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()

            batch_size = action_labels.size(0)
            loss_sum += loss.detach().item() * batch_size

            action_preds = torch.argmax(action_logits, dim=1)
            action_correct += (action_preds == action_labels).sum().item()
            action_total += batch_size

            aggiorna_confusion_matrix(action_conf_matrix, action_labels, action_preds)

            if shot_mask.any():
                outcome_preds = torch.argmax(outcome_logits[shot_mask], dim=1)
                outcome_targets = canestro[shot_mask]
                outcome_correct += (outcome_preds == outcome_targets).sum().item()
                outcome_total += outcome_targets.size(0)
                aggiorna_confusion_matrix(outcome_conf_matrix, outcome_targets, outcome_preds)

    mean_loss = loss_sum / max(action_total, 1)
    action_accuracy = action_correct / max(action_total, 1)
    outcome_accuracy = outcome_correct / max(outcome_total, 1)

    action_recalls = recall_per_classe(action_conf_matrix)
    action_valid = action_conf_matrix.sum(dim=1) > 0
    action_macro_recall = action_recalls[action_valid].mean().item() if action_valid.any() else 0.0

    outcome_recalls = recall_per_classe(outcome_conf_matrix)
    outcome_valid = outcome_conf_matrix.sum(dim=1) > 0
    outcome_balanced_accuracy = outcome_recalls[outcome_valid].mean().item() if outcome_valid.any() else 0.0

    return {
        "loss": mean_loss,
        "action_accuracy": action_accuracy,
        "outcome_accuracy": outcome_accuracy,
        "action_macro_recall": action_macro_recall,
        "outcome_balanced_accuracy": outcome_balanced_accuracy,
        "action_conf_matrix": action_conf_matrix,
        "outcome_conf_matrix": outcome_conf_matrix,
    }


# ============================================================
# TRAINING
# ============================================================

best_val_score = float("-inf")
epochs_without_improvement = 0

for epoch in range(1, NUM_EPOCHS + 1):
    train_metrics = esegui_epoca(train_dataloader, training=True)
    val_metrics = esegui_epoca(val_dataloader, training=False)

    val_score = (
        0.5 * val_metrics["action_macro_recall"]
        + 0.5 * val_metrics["outcome_balanced_accuracy"]
    )

    scheduler.step(val_score)
    current_lr = optimizer.param_groups[0]["lr"]

    print(f"\nEpoch [{epoch}/{NUM_EPOCHS}] - LR: {current_lr:.6g}")
    print(f"Train Loss: {train_metrics['loss']:.4f}")
    print(f"Train Action Acc: {train_metrics['action_accuracy']:.4f} | Macro Recall: {train_metrics['action_macro_recall']:.4f}")
    print(f"Train Outcome Acc: {train_metrics['outcome_accuracy']:.4f} | Balanced Acc: {train_metrics['outcome_balanced_accuracy']:.4f}")
    
    print(f"Val Loss: {val_metrics['loss']:.4f}")
    print(f"Val Action Acc: {val_metrics['action_accuracy']:.4f} | Macro Recall: {val_metrics['action_macro_recall']:.4f}")
    print(f"Val Outcome Acc: {val_metrics['outcome_accuracy']:.4f} | Balanced Acc: {val_metrics['outcome_balanced_accuracy']:.4f}")
    print(f"Val Score: {val_score:.4f}")

    stampa_metriche_classi(val_metrics["action_conf_matrix"], train_dataset.idx_to_action, "Confusion Matrix Validation - Azione")
    stampa_metriche_classi(val_metrics["outcome_conf_matrix"], train_dataset.idx_to_outcome, "Confusion Matrix Validation - Esito")

    if val_score > best_val_score:
        best_val_score = val_score
        epochs_without_improvement = 0

        torch.save(
            {
                "model_type": "MoViNet_RFDETR_LiveFusion_MultiTask",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
                "val_score": val_score,
                "val_metrics": {
                    key: value for key, value in val_metrics.items() if "conf_matrix" not in key
                },
                "action_to_idx": train_dataset.action_to_idx,
                "idx_to_action": train_dataset.idx_to_action,
                "outcome_to_idx": train_dataset.outcome_to_idx,
                "idx_to_outcome": train_dataset.idx_to_outcome,
                "movinet_size": MOVINET_FEATURE_DIM,
                "rfdetr_size": RFDETR_FEATURE_DIM,
                "rfdetr_encoded_size": RFDETR_ENCODED_DIM,
                "hidden_size": HIDDEN_SIZE,
                "num_action_classes": num_action_classes,
                "max_frames": MAX_FRAMES,
                "img_size": IMG_SIZE,
                "action_weights": action_weights.detach().cpu(),
                "outcome_weights": outcome_weights.detach().cpu(),
            },
            CHECKPOINT_PATH,
        )
        print(f"Nuovo miglior modello salvato in: {CHECKPOINT_PATH}")
    else:
        epochs_without_improvement += 1
        print(f"Nessun miglioramento: {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE}")
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("Early stopping.")
            break

# ============================================================
# TEST DEL MIGLIOR CHECKPOINT
# ============================================================

if not Path(CHECKPOINT_PATH).exists():
    raise FileNotFoundError(f"Il miglior checkpoint non è stato creato: {CHECKPOINT_PATH}")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False,
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

test_metrics = esegui_epoca(test_dataloader, training=False)

print("\n==============================")
print("RISULTATI FINALI SUL TEST SET")
print("==============================")
print(f"Test Loss: {test_metrics['loss']:.4f}")
print(f"Test Action Acc: {test_metrics['action_accuracy']:.4f} | Macro Recall: {test_metrics['action_macro_recall']:.4f}")
print(f"Test Outcome Acc: {test_metrics['outcome_accuracy']:.4f} | Balanced Acc: {test_metrics['outcome_balanced_accuracy']:.4f}")

stampa_metriche_classi(test_metrics["action_conf_matrix"], test_dataset.idx_to_action, "Confusion Matrix Test - Azione")
stampa_metriche_classi(test_metrics["outcome_conf_matrix"], test_dataset.idx_to_outcome, "Confusion Matrix Test - Esito")