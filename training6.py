import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Import dei vostri moduli personalizzati
from dataset import VideoDataset
from BasketMoViNetMultitask import BasketMoViNetMultitask

# ============================================================================
# 1. CONFIGURAZIONE PERCORSI E IPERPARAMETRI
# ============================================================================
FILE_ATTUALE = os.path.dirname(os.path.abspath(__file__))
DATASET_CARTELLA = os.path.abspath(os.path.join(FILE_ATTUALE, "..", "dataset"))
MANIFEST_PATH = os.path.join(DATASET_CARTELLA, "manifest.csv")
CACHE_FRAMES_DIR = os.path.join(DATASET_CARTELLA, "video_32_frame")
RFDETR_FEATURES_DIR = os.path.join(DATASET_CARTELLA, "rfdetr_features")

# Iperparametri di addestramento
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4  # Learning rate conservativo per il fine-tuning di MoViNet
MAX_FRAME = 32
IMG_SIZE = 224  # Risoluzione standard nativa per MoViNet-A0
NUM_ACTION_CLASSES = 5  # Adatta questo numero al tuo effettivo numero di classi
RFDETR_FEATURE_DIM = 19

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Utilizzo del dispositivo: {DEVICE}")

# ============================================================================
# 2. UTILITY PER METRICHE (CONFUSION MATRIX)
# ============================================================================
def aggiorna_confusion_matrix(conf_matrix, labels, preds, ignore_idx=None):
    labels = labels.cpu()
    preds = preds.cpu()
    for true_label, pred_label in zip(labels, preds):
        if ignore_idx is not None and true_label == ignore_idx:
            continue
        conf_matrix[true_label, pred_label] += 1
    return conf_matrix

def stampa_confusion_matrix(conf_matrix, idx_to_class, titolo="Confusion Matrix"):
    print(f"\n🔹 {titolo}")
    print("Righe = classe vera | Colonne = classe predetta\n")
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    header = "vera\\pred".ljust(18)
    for name in class_names:
        header += name[:10].ljust(12)
    print(header)
    
    for i in range(len(class_names)):
        riga = class_names[i][:15].ljust(18)
        for j in range(len(class_names)):
            riga += str(int(conf_matrix[i, j])).ljust(12)
        print(riga)

# ============================================================================
# 3. PREPARAZIONE DATASET E DATALOADER
# ============================================================================
# Trasformazione standard per MoViNet pre-addestrata su Kinetics (ImageNet stats)
transform_video = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

print("📦 Caricamento dei Dataset...")
train_dataset = VideoDataset(
    manifest_path=MANIFEST_PATH,
    video_dir=DATASET_CARTELLA,
    split="train",
    maxFrame=MAX_FRAME,
    imgSize=IMG_SIZE,
    transform=transform_video,
    cache_dir=CACHE_FRAMES_DIR,
    rfdetr_features_dir=RFDETR_FEATURES_DIR,
    rfdetr_feature_dim=RFDETR_FEATURE_DIM
)

val_dataset = VideoDataset(
    manifest_path=MANIFEST_PATH,
    video_dir=DATASET_CARTELLA,
    split="val",  # Assicurati che si chiami "val" o "validation" nel manifest
    maxFrame=MAX_FRAME,
    imgSize=IMG_SIZE,
    transform=transform_video,
    cache_dir=CACHE_FRAMES_DIR,
    rfdetr_features_dir=RFDETR_FEATURES_DIR,
    rfdetr_feature_dim=RFDETR_FEATURE_DIM
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ============================================================================
# 4. INIZIALIZZAZIONE MODELLO, COSTRUTTORI DI LOSS E OTTIMIZZATORE
# ============================================================================
model = BasketMoViNetMultitask(
    num_action_classes=NUM_ACTION_CLASSES, 
    rfdetr_feature_dim=RFDETR_FEATURE_DIM
)

# Gestione Multi-GPU se disponibile (ereditata dal vostro training5.py)
if torch.cuda.device_count() > 1:
    print(f"🔥 Rilevate {torch.cuda.device_count()} GPU. Attivo DataParallel.")
    model = nn.DataParallel(model)

model = model.to(DEVICE)

# Loss functions
criterion_action = nn.CrossEntropyLoss()
# CRUCIAL: ignore_index=-1 evita che i passaggi/idle influiscano sulla loss dell'esito tiro
criterion_outcome = nn.CrossEntropyLoss(ignore_index=-1)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# ============================================================================
# 5. LOOP DI ADDESTRAMENTO
# ============================================================================
best_val_score = 0.0

for epoch in range(EPOCHS):
    print(f"\n🎬 --- EPOCA {epoch + 1}/{EPOCHS} ---")
    
    # ------------------ FASE DI TRAIN ------------------
    model.train()
    running_loss = 0.0
    correct_action = 0
    correct_outcome = 0
    total_samples = 0
    total_outcome_samples = 0
    
    # Il vostro dataset restituisce: video, rfdetr, action, outcome, mask, is_shot
    for videos, rfdetr_feats, action_labels, outcome_labels, _, _ in tqdm(train_loader, desc="Training"):
        videos = videos.to(DEVICE)          # [B, F, C, H, W]
        rfdetr_feats = rfdetr_feats.to(DEVICE)  # [B, F, 19]
        action_labels = action_labels.to(DEVICE)
        outcome_labels = outcome_labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Forward pass (MoViNet si occupa della permutazione interna)
        out_action, out_outcome = model(videos, rfdetr_feats)
        
        # Calcolo delle Loss individuali
        loss_act = criterion_action(out_action, action_labels)
        loss_out = criterion_outcome(out_outcome, outcome_labels)
        loss_total = loss_act + loss_out
        
        loss_total.backward()
        optimizer.step()
        
        # Calcolo Metriche Train
        running_loss += loss_total.item() * videos.size(0)
        total_samples += videos.size(0)
        
        # Accuratezza Azione
        _, pred_action = torch.max(out_action, 1)
        correct_action += (pred_action == action_labels).sum().item()
        
        # Accuratezza Esito (solo sui tiri reali, escludendo i -1)
        _, pred_outcome = torch.max(out_outcome, 1)
        mask_shot = (outcome_labels != -1)
        correct_outcome += ((pred_outcome == outcome_labels) & mask_shot).sum().item()
        total_outcome_samples += mask_shot.sum().item()

    train_loss = running_loss / total_samples
    train_action_acc = (correct_action / total_samples) * 100
    train_outcome_acc = (correct_outcome / total_outcome_samples) * 100 if total_outcome_samples > 0 else 0.0
    
    print(f"📊 Train Loss: {train_loss:.4f} | Action Acc: {train_action_acc:.2f}% | Outcome Acc: {train_outcome_acc:.2f}%")

    # ------------------ FASE DI VALIDATION ------------------
    model.eval()
    val_running_loss = 0.0
    val_correct_action = 0
    val_correct_outcome = 0
    val_total_samples = 0
    val_total_outcome_samples = 0
    
    # Inizializzazione matrici di confusione
    action_conf_matrix = torch.zeros(NUM_ACTION_CLASSES, NUM_ACTION_CLASSES)
    outcome_conf_matrix = torch.zeros(2, 2)
    
    with torch.no_grad():
        for videos, rfdetr_feats, action_labels, outcome_labels, _, _ in val_loader:
            videos = videos.to(DEVICE)
            rfdetr_feats = rfdetr_feats.to(DEVICE)
            action_labels = action_labels.to(DEVICE)
            outcome_labels = outcome_labels.to(DEVICE)
            
            out_action, out_outcome = model(videos, rfdetr_feats)
            
            loss_act = criterion_action(out_action, action_labels)
            loss_out = criterion_outcome(out_outcome, outcome_labels)
            val_loss_total = loss_act + loss_out
            
            val_running_loss += val_loss_total.item() * videos.size(0)
            val_total_samples += videos.size(0)
            
            # Predizioni e matrici
            _, pred_action = torch.max(out_action, 1)
            val_correct_action += (pred_action == action_labels).sum().item()
            aggiorna_confusion_matrix(action_conf_matrix, action_labels, pred_action)
            
            _, pred_outcome = torch.max(out_outcome, 1)
            mask_shot = (outcome_labels != -1)
            val_correct_outcome += ((pred_outcome == outcome_labels) & mask_shot).sum().item()
            val_total_outcome_samples += mask_shot.sum().item()
            aggiorna_confusion_matrix(outcome_conf_matrix, outcome_labels, pred_outcome, ignore_idx=-1)

    val_loss = val_running_loss / val_total_samples
    val_action_acc = (val_correct_action / val_total_samples) * 100
    val_outcome_acc = (val_correct_outcome / val_total_outcome_samples) * 100 if val_total_outcome_samples > 0 else 0.0
    
    # Score combinato per il salvataggio del modello migliore (Media pesata o aritmetica)
    val_score = (val_action_acc + val_outcome_acc) / 2
    
    print(f"🔬 Val Loss: {val_loss:.4f} | Val Action Acc: {val_action_acc:.2f}% | Val Outcome Acc: {val_outcome_acc:.2f}%")
    
    # Stampa dei log di classificazione a schermo
    stampa_confusion_matrix(action_conf_matrix, train_dataset.idx_to_action, "Confusion Matrix Validation - Azione")
    stampa_confusion_matrix(outcome_conf_matrix, {0: "Sbagliato", 1: "Segnato"}, "Confusion Matrix Validation - Esito Tiro")

    # ============================================================================
    # 6. SALVATAGGIO DEL MIGLIOR MODELLO
    # ============================================================================
    if val_score > best_val_score:
        best_val_score = val_score
        print(f"🏆 Nuovo miglior punteggio di Validation ({val_score:.2f}%)! Salvataggio checkpoint...")
        
        # Estrae il modello liscio se si usa DataParallel
        model_to_save = model.module if isinstance(model, nn.DataParallel) else model
        
        checkpoint_path = os.path.join(DATASET_CARTELLA, "movinet_multitask_best.pth")
        torch.save({
            "model_type": "BasketMoViNetMultitask",
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "val_score": val_score,
            "val_action_acc": val_action_acc,
            "val_outcome_acc": val_outcome_acc,
            "action_to_idx": train_dataset.action_to_idx,
            "idx_to_action": train_dataset.idx_to_action,
        }, checkpoint_path)
        print(f"💾 Checkpoint salvato in: {checkpoint_path}")

print("\n🏁 Addestramento completato!")