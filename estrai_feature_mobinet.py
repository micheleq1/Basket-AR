import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Assicurati di aver installato: pip install git+https://github.com/Atze00/MoViNet-pytorch.git
from movinets import MoViNet
from movinets.config import _C

# CONFIGURAZIONE DEI PERCORSI
FILE_ATTUALE = os.path.dirname(os.path.abspath(__file__))
DATASET_CARTELLA = os.path.abspath(os.path.join(FILE_ATTUALE, "..", "dataset"))
MANIFEST = os.path.join(DATASET_CARTELLA, "manifest.csv")
CACHE_FRAMES = os.path.join(DATASET_CARTELLA, "video_32_frame")
OUTPUT_MOVINET_DIR = os.path.join(DATASET_CARTELLA, "movinet_features")

os.makedirs(OUTPUT_MOVINET_DIR, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Uso il device per l'estrazione: {device}")

# 1. Caricamento MoViNet-A2 ed eliminazione del modulo di classificazione (Kinetics)
print("Inizializzazione MoViNet-A2 preaddestrato...")
movinet = MoViNet(_C.MODEL.MoViNetA2, causal=False, pretrained=True)
# Sostituiamo il classificatore finale con un'identità per estrarre le feature pure
movinet.classifier = nn.Identity()
movinet = movinet.to(device)
movinet.eval()

# 2. Lettura del manifest
df = pd.read_csv(MANIFEST)
print(f"Video totali da processare: {len(df)}")

# Estrazione senza calcolo dei gradienti (congelato)
with torch.no_grad():
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Estrazione Feature Video"):
        path = row["path"]
        nome_file = path.replace("/", "_").replace("\\", "_") + ".npy"
        
        percorso_input_frames = os.path.join(CACHE_FRAMES, nome_file)
        percorso_output_feat = os.path.join(OUTPUT_MOVINET_DIR, nome_file)
        
        # Salta se il file è già stato estratto (comodo se si interrompe lo script)
        if os.path.exists(percorso_output_feat):
            continue
            
        if not os.path.exists(percorso_input_frames):
            print(f"\n[ATTENZIONE] Cache frame mancante per: {path}")
            continue
            
        # Carica i frame salvati (shape originale: [32, 704, 704, 3], uint8)
        frames = np.load(percorso_input_frames)
        
        # Convertiamo in tensore PyTorch e normalizziamo: [T, H, W, C] -> [T, C, H, W]
        tensor_frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        
        # MoViNet-A2 predilige la risoluzione standard 224x224. Ridimensioniamo nello spazio:
        tensor_frames = F.interpolate(tensor_frames, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Formato richiesto da MoViNet: [B, C, T, H, W]
        # Spostiamo il canale colore prima del tempo e aggiungiamo la dimensione del Batch (B=1)
        tensor_frames = tensor_frames.permute(1, 0, 2, 3).unsqueeze(0).to(device)
        
        # Forward pass: MoViNet applica un Global Spatio-Temporal Average Pooling nativo
        # Output atteso per MoViNet-A2: [1, 480]
        features = movinet(tensor_frames)
        features_np = features.squeeze(0).cpu().numpy() # Portiamo a vettore 1D di 480 elementi
        
        # Salva le feature spazio-temporali su disco
        np.save(percorso_output_feat, features_np)

print(f"\nEstrazione completata! Feature salvate in: {OUTPUT_MOVINET_DIR}")