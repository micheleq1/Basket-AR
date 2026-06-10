import pandas as pd
import os
import numpy as np
from video_preprocessor import VideoPreprocessor


FILE_ATTUALE = os.path.dirname(os.path.abspath(__file__))

DATASET_CARTELLA = os.path.abspath(
    os.path.join(FILE_ATTUALE, "..", "dataset")
)

MANIFEST = os.path.abspath(
    os.path.join(DATASET_CARTELLA, "manifest.csv")
)

df = pd.read_csv(MANIFEST)


MAX_FRAME = 32
IMG_SIZE = 704

video_preprocessor = VideoPreprocessor(
    max_frame=MAX_FRAME,
    img_size=IMG_SIZE
)

CACHE_FRAMES = os.path.abspath(
    os.path.join(DATASET_CARTELLA, "video_32_frame")
)

MASK_FRAMES = os.path.abspath(
    os.path.join(DATASET_CARTELLA, "mask_frame")
)

os.makedirs(CACHE_FRAMES, exist_ok=True)
os.makedirs(MASK_FRAMES, exist_ok=True)


print("Manifest:", MANIFEST)
print("Dataset:", DATASET_CARTELLA)
print("Output frame:", CACHE_FRAMES)
print("Output mask:", MASK_FRAMES)
print("Video totali da processare:", len(df))


for _, row in df.iterrows():

    path = row["path"]
    split = row["split"]

    percorso_video = os.path.abspath(
        os.path.join(DATASET_CARTELLA, path)
    )

    nome_file = path.replace("/", "_").replace("\\", "_") + ".npy"

    percorso_output = os.path.join(
        CACHE_FRAMES,
        nome_file
    )

    percorso_mask = os.path.join(
        MASK_FRAMES,
        nome_file.replace(".npy", "_mask.npy")
    )

    if os.path.exists(percorso_output) and os.path.exists(percorso_mask):
        print(f"Già esistente, salto: [{split}] {path}")
        continue

    if not os.path.exists(percorso_video):
        print(f"Video non trovato, salto: {percorso_video}")
        continue

    try:
        frames, mask, total_frames = video_preprocessor(percorso_video)

    except ValueError as errore:
        print(f"Errore su {path}: {errore}")
        continue

    np.save(percorso_output, frames)
    np.save(percorso_mask, mask)

    print(f"Salvato: [{split}] {path}")
    print(f"  Frame originali: {total_frames}")
    print(f"  Shape frames: {frames.shape}")
    print(f"  Shape mask: {mask.shape}")