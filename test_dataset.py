import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from dataset import VideoDataset

# 1. Recuperiamo i percorsi delle cartelle
FILE_ATTUALE = os.path.dirname(os.path.abspath(__file__))
DATASET_CARTELLA = os.path.abspath(os.path.join(FILE_ATTUALE, "..", "dataset"))

MANIFEST = os.path.abspath(os.path.join(DATASET_CARTELLA, "manifest.csv"))
CACHE_FRAMES = os.path.abspath(os.path.join(DATASET_CARTELLA, "video_32_frame"))
MASK_FRAMES = os.path.abspath(os.path.join(DATASET_CARTELLA, "mask_frame"))

# Creiamo una trasformazione che "non fa nulla" (Identity)
# Serve solo ad attivare il blocco `if self.transform:` dentro il dataset
trasformazione_test = transforms.Lambda(lambda x: x)

# Inizializziamo il dataset in modalità train con la trasformazione di test

dataset_train = VideoDataset(
    manifest_path=MANIFEST,
    video_dir=DATASET_CARTELLA,
    cache_dir=CACHE_FRAMES,
    mask_dir=MASK_FRAMES,
    split="val",
    maxFrame=32,
    imgSize=224,
    transform=trasformazione_test  # <--- Attiva l'augmentation!
)

print("Classi rare individuate dal dataset:", dataset_train.classi_rare)

# 2. Cerchiamo automaticamente il primo video che appartiene a una classe rara
indice_video_raro = None
for i in range(len(dataset_train)):
    label_name = dataset_train.video_split.iloc[i, 5]
    if label_name in dataset_train.classi_rare:
        indice_video_raro = i
        print(f"\nTrovato video aumentabile all'indice {i} | Classe originale: {label_name}")
        break

if indice_video_raro is None:
    print("\n[ATTENZIONE] Nessun video di classe rara trovato. Uso l'indice 0 per il test.")
    indice_video_raro = 0

# 3. Estraiamo lo STESSO identico video due volte di seguito.
# Poiché l'augmentation è casuale (ha un 50% di probabilità di flip e una luminosità randomica),
# i due caricamenti consecutivi dovrebbero mostrare differenze!
frames_1, _, _, _, _ = dataset_train[indice_video_raro]
frames_2, _, _, _, _ = dataset_train[indice_video_raro]

# Prendiamo il primo frame del primo caricamento e del secondo caricamento
img_tentativo_1 = frames_1[0].permute(1, 2, 0).numpy()
img_tentativo_2 = frames_2[0].permute(1, 2, 0).numpy()

# 4. Mostriamo il confronto a schermo
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(img_tentativo_1)
axes[0].set_title("Estratto 1 (Casuale)")
axes[0].axis('off')

axes[1].imshow(img_tentativo_2)
axes[1].set_title("Estratto 2 (Casuale)")
axes[1].axis('off')

plt.suptitle(f"Verifica Data Augmentation - Indice Video: {indice_video_raro}", fontsize=14)
plt.tight_layout()
plt.show()