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
train = df[df["split"] == "train"]


MAX_FRAME = 32
IMG_SIZE = 224

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

for _, row in train.iterrows():

    path = row["path"]

    percorso_video = os.path.abspath(
        os.path.join(DATASET_CARTELLA, path)
    )

    nome_file = path.replace("/", "_").replace("\\", "_") + ".npy"

    percorso_output = os.path.join(CACHE_FRAMES, nome_file)

    percorso_mask = os.path.join(
        MASK_FRAMES,
        nome_file.replace(".npy", "_mask.npy")
    )

    if os.path.exists(percorso_output) and os.path.exists(percorso_mask):
        print(f"Già esistente, salto: {path}")
        continue

    try:
        frames, mask, total_frames = video_preprocessor(percorso_video)

    except ValueError as errore:
        print(errore)
        continue

    np.save(percorso_output, frames)
    np.save(percorso_mask, mask)

    #print(f"Salvato: {path}")
    #print(f"Numero frame originali: {total_frames}")
    #print(f"Shape finale frames: {frames.shape}")
    #print(f"Shape mask: {mask.shape}")
    #print(f"Mask: {mask}")