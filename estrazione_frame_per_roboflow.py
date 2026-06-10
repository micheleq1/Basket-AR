import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from rfdetr import RFDETRLarge


# ============================================================
# PERCORSI
# ============================================================

FILE_ATTUALE = os.path.dirname(os.path.abspath(__file__))

DATASET_CARTELLA = os.path.abspath(
    os.path.join(FILE_ATTUALE, "..", "dataset")
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

CHECKPOINT_PATH = os.path.abspath(
    os.path.join(DATASET_CARTELLA, "checkpoint_best_total.pth")
)


# ============================================================
# CONFIGURAZIONE RF-DETR
# ============================================================

INPUT_SIZE = 704
CONF_THRESHOLD = 0.40

# Modifica solo se le classi sono invertite
CLASS_ID_PALLA = 0
CLASS_ID_CANESTRO = 1

# Se i frame salvati sono RGB lascia True.
# Se invece li hai salvati direttamente con OpenCV, potrebbero essere BGR: metti False.
FRAMES_ARE_RGB = True


# ============================================================
# PREPARAZIONE FRAME
# ============================================================

def prepara_frame(frame):
    """
    Porta il frame nel formato corretto per RF-DETR:
    - formato H, W, C
    - tipo uint8
    - colore RGB
    """

    frame = np.asarray(frame)

    # Caso frame in formato C, H, W
    if frame.ndim == 3 and frame.shape[0] == 3:
        frame = np.transpose(frame, (1, 2, 0))

    # Caso frame float 0-1 oppure float 0-255
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = frame * 255.0

        frame = np.clip(frame, 0, 255).astype(np.uint8)

    # Se i frame sono BGR, converto in RGB
    if not FRAMES_ARE_RGB:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame


# ============================================================
# ESTRAZIONE DETECTION
# ============================================================

def prendi_detection_migliore(detections, class_id_target):
    """
    Tra tutte le detection, prende quella con confidence più alta
    per la classe richiesta.
    """

    if detections is None or len(detections) == 0:
        return None, 0.0

    best_box = None
    best_conf = 0.0

    for box, class_id, conf in zip(
        detections.xyxy,
        detections.class_id,
        detections.confidence
    ):
        if int(class_id) == class_id_target:
            conf = float(conf)

            if conf > best_conf:
                best_conf = conf
                best_box = box

    return best_box, best_conf


def box_to_feature(box, conf, frame_w, frame_h):
    """
    Converte una bounding box in feature normalizzate.

    Output:
    [
        presente,
        centro_x,
        centro_y,
        larghezza,
        altezza,
        confidence
    ]
    """

    if box is None:
        return [
            0.0,  # presente
            0.0,  # x
            0.0,  # y
            0.0,  # w
            0.0,  # h
            0.0   # confidence
        ]

    x1, y1, x2, y2 = box

    centro_x = ((x1 + x2) / 2.0) / frame_w
    centro_y = ((y1 + y2) / 2.0) / frame_h

    larghezza = (x2 - x1) / frame_w
    altezza = (y2 - y1) / frame_h

    return [
        1.0,
        float(centro_x),
        float(centro_y),
        float(larghezza),
        float(altezza),
        float(conf)
    ]


def estrai_feature_frame(frame_rgb, detections):
    """
    Estrae le feature geometriche da un singolo frame.

    Feature prodotte:

    0  palla_presente
    1  palla_x
    2  palla_y
    3  palla_w
    4  palla_h
    5  palla_conf

    6  canestro_presente
    7  canestro_x
    8  canestro_y
    9  canestro_w
    10 canestro_h
    11 canestro_conf

    12 dx_palla_canestro
    13 dy_palla_canestro
    14 distanza_palla_canestro
    """

    frame_h, frame_w = frame_rgb.shape[:2]

    palla_box, palla_conf = prendi_detection_migliore(
        detections,
        CLASS_ID_PALLA
    )

    canestro_box, canestro_conf = prendi_detection_migliore(
        detections,
        CLASS_ID_CANESTRO
    )

    palla_feature = box_to_feature(
        palla_box,
        palla_conf,
        frame_w,
        frame_h
    )

    canestro_feature = box_to_feature(
        canestro_box,
        canestro_conf,
        frame_w,
        frame_h
    )

    palla_presente = palla_feature[0]
    canestro_presente = canestro_feature[0]

    palla_x = palla_feature[1]
    palla_y = palla_feature[2]

    canestro_x = canestro_feature[1]
    canestro_y = canestro_feature[2]

    if palla_presente == 1.0 and canestro_presente == 1.0:
        dx = palla_x - canestro_x
        dy = palla_y - canestro_y
        distanza = np.sqrt(dx ** 2 + dy ** 2)
    else:
        dx = 0.0
        dy = 0.0
        distanza = 0.0

    feature_finale = (
        palla_feature
        + canestro_feature
        + [
            float(dx),
            float(dy),
            float(distanza)
        ]
    )

    return np.array(feature_finale, dtype=np.float32)


# ============================================================
# VELOCITÀ PALLA
# ============================================================

def aggiungi_velocita_palla(sequence):
    """
    Aggiunge 4 feature:

    15 velocita_x_palla
    16 velocita_y_palla
    17 velocita_palla
    18 velocita_valida

    Input:
    sequence shape = (32, 15)

    Output:
    sequence shape = (32, 19)
    """

    num_frames = sequence.shape[0]

    velocita = np.zeros((num_frames, 4), dtype=np.float32)

    for i in range(1, num_frames):
        palla_presente_ora = sequence[i, 0]
        palla_presente_prima = sequence[i - 1, 0]

        if palla_presente_ora == 1.0 and palla_presente_prima == 1.0:
            x_ora = sequence[i, 1]
            y_ora = sequence[i, 2]

            x_prima = sequence[i - 1, 1]
            y_prima = sequence[i - 1, 2]

            vx = x_ora - x_prima
            vy = y_ora - y_prima
            speed = np.sqrt(vx ** 2 + vy ** 2)

            velocita[i, 0] = vx
            velocita[i, 1] = vy
            velocita[i, 2] = speed
            velocita[i, 3] = 1.0

    sequence = np.concatenate(
        [sequence, velocita],
        axis=1
    )

    return sequence.astype(np.float32)


# ============================================================
# PROCESSO SINGOLO VIDEO
# ============================================================

def processa_video(frames_path, mask_path, output_path, model):
    frames = np.load(frames_path)
    mask = np.load(mask_path).reshape(-1)

    if len(frames) != len(mask):
        raise ValueError(
            f"Errore dimensioni su {frames_path.name}: "
            f"{len(frames)} frame, {len(mask)} valori mask"
        )

    feature_video = []

    for frame, mask_value in zip(frames, mask):

        # Frame di padding: non faccio inference
        if int(mask_value) == 0:
            feature_video.append(
                np.zeros(15, dtype=np.float32)
            )
            continue

        frame_rgb = prepara_frame(frame)

        detections = model.predict(
            frame_rgb,
            threshold=CONF_THRESHOLD,
            shape=(INPUT_SIZE, INPUT_SIZE),
            include_source_image=False
        )

        feature_frame = estrai_feature_frame(
            frame_rgb,
            detections
        )

        feature_video.append(feature_frame)

    feature_video = np.stack(feature_video, axis=0)

    feature_video = aggiungi_velocita_palla(feature_video)

    np.save(output_path, feature_video)


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(RFDETR_FEATURES, exist_ok=True)

    if not Path(CACHE_FRAMES).exists():
        raise FileNotFoundError(f"Cartella frame non trovata: {CACHE_FRAMES}")

    if not Path(MASK_FRAMES).exists():
        raise FileNotFoundError(f"Cartella mask non trovata: {MASK_FRAMES}")

    if not Path(CHECKPOINT_PATH).exists():
        raise FileNotFoundError(f"Checkpoint RF-DETR non trovato: {CHECKPOINT_PATH}")

    print("Carico modello RF-DETR...")
    print(CHECKPOINT_PATH)

    model = RFDETRLarge(
        pretrain_weights=CHECKPOINT_PATH,
        num_classes=2
    )

    model.optimize_for_inference()

    frame_files = sorted(Path(CACHE_FRAMES).glob("*.npy"))

    print(f"\nVideo trovati: {len(frame_files)}")
    print(f"Feature RF-DETR salvate in: {RFDETR_FEATURES}\n")

    for frames_path in tqdm(frame_files):
        mask_path = Path(MASK_FRAMES) / frames_path.name
        output_path = Path(RFDETR_FEATURES) / frames_path.name

        if output_path.exists():
            continue

        if not mask_path.exists():
            print(f"Mask mancante per: {frames_path.name}")
            continue

        try:
            processa_video(
                frames_path=frames_path,
                mask_path=mask_path,
                output_path=output_path,
                model=model
            )

        except Exception as e:
            print(f"Errore su {frames_path.name}: {e}")

    print("\nEstrazione feature RF-DETR completata.")


if __name__ == "__main__":
    main()