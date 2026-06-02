import cv2
import pandas as pd
import os

FILE_ATTUALE = os.path.dirname(os.path.abspath(__file__))

DATASET_CARTELLA = os.path.abspath(
     os.path.join(FILE_ATTUALE, "..", "dataset")
)
MANIFEST = os.path.abspath(
     os.path.join(DATASET_CARTELLA, "manifest.csv")
 )

manifest = pd.read_csv(MANIFEST)

durate = []

for i, row in manifest.iterrows():
    rel_path = row.iloc[1]
    label = row.iloc[5]

    video_path = os.path.join(DATASET_CARTELLA, rel_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Errore apertura:", video_path)
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    durata = total_frames / fps if fps > 0 else 0

    durate.append({
        "video": rel_path,
        "label": label,
        "fps": fps,
        "total_frames": total_frames,
        "durata_secondi": durata
    })

df = pd.DataFrame(durate)

print(df["durata_secondi"].describe())

print("\nDurata media per classe:")
print(df.groupby("label")["durata_secondi"].describe())