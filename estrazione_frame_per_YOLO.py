import pandas as pd
import os
import cv2
import csv


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


# ==========================
# PARAMETRI
# ==========================

MAX_FRAME = 32

YOLO_CARTELLA = os.path.abspath(
    os.path.join(DATASET_CARTELLA, "yolo_frames_ball")
)

IMAGES_DA_ANNOTARE = os.path.join(
    YOLO_CARTELLA,
    "images_da_annotare"
)

MAPPING_CSV = os.path.join(
    YOLO_CARTELLA,
    "mapping_frame_video.csv"
)

os.makedirs(IMAGES_DA_ANNOTARE, exist_ok=True)


# ==========================
# FUNZIONE INDICI FRAME
# ==========================

def crea_indici_primi_ultimi(total_frames, max_frame):
    """
    Se il video ha almeno max_frame frame:
    prende max_frame/2 frame iniziali
    e max_frame/2 frame finali.

    Se il video ha meno di max_frame frame:
    prende tutti i frame reali disponibili.

    Per YOLO non vengono creati frame nulli.
    """

    if total_frames <= max_frame:
        return list(range(total_frames))

    n_first = max_frame // 2
    n_last = max_frame - n_first

    primi = list(range(0, n_first))

    start_last = total_frames - n_last
    ultimi = list(range(start_last, total_frames))

    return primi + ultimi


# ==========================
# CARICAMENTO MANIFEST
# ==========================

df = pd.read_csv(MANIFEST)

# Uso solo lo split train per evitare data leakage
train = df[df["split"] == "train"].reset_index(drop=True)


# ==========================
# ESTRAZIONE FRAME
# ==========================

righe_mapping = []

for video_idx, row in train.iterrows():

    path = row["path"]

    # Provo a leggere la label dal manifest.
    # Se non esiste la colonna "label", uso la sesta colonna come nel tuo Dataset.
    if "label" in row.index:
        label = row["label"]
    else:
        label = row.iloc[5]

    percorso_video = os.path.abspath(
        os.path.join(DATASET_CARTELLA, path)
    )

    cap = cv2.VideoCapture(percorso_video)

    if not cap.isOpened():
        print(f"Errore apertura video: {percorso_video}")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        print(f"Video senza frame validi: {percorso_video}")
        cap.release()
        continue

    indici_da_salvare = crea_indici_primi_ultimi(
        total_frames=total_frames,
        max_frame=MAX_FRAME
    )

    indici_set = set(indici_da_salvare)

    # Creo la cartella della classe
    cartella_classe = os.path.join(
        IMAGES_DA_ANNOTARE,
        label
    )

    os.makedirs(cartella_classe, exist_ok=True)

    # Creo un nome base pulito partendo dal path del video
    nome_base = path.replace("/", "_").replace("\\", "_").replace(".mp4", "")

    current_index = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if current_index in indici_set:

            nome_immagine = f"{nome_base}_frame_{current_index:06d}.jpg"

            percorso_output = os.path.join(
                cartella_classe,
                nome_immagine
            )

            # Non ridimensioniamo.
            # cv2 legge in BGR e cv2.imwrite salva correttamente in BGR.
            cv2.imwrite(percorso_output, frame)

            righe_mapping.append({
                "image_name": nome_immagine,
                "class_folder": label,
                "image_path": os.path.join(label, nome_immagine),
                "video_path": path,
                "label": label,
                "frame_index": current_index,
                "total_frames_video": total_frames
            })

            saved_count += 1

        current_index += 1

    cap.release()

    print(
        f"Video {video_idx + 1}/{len(train)} | "
        f"Classe: {label} | "
        f"Frame salvati: {saved_count} | "
        f"Video: {path}"
    )


# ==========================
# SALVATAGGIO MAPPING CSV
# ==========================

with open(MAPPING_CSV, mode="w", newline="", encoding="utf-8") as file_csv:
    fieldnames = [
        "image_name",
        "class_folder",
        "image_path",
        "video_path",
        "label",
        "frame_index",
        "total_frames_video"
    ]

    writer = csv.DictWriter(file_csv, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(righe_mapping)


print("\nEstrazione completata.")
print(f"Immagini salvate in: {IMAGES_DA_ANNOTARE}")
print(f"Mapping salvato in: {MAPPING_CSV}")