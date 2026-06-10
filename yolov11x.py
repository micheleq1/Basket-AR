from pathlib import Path
import torch
from ultralytics import YOLO
import os

# ============================================================
# MODIFICA SOLO QUESTO PERCORSO
# ============================================================
FILE_ATTUALE = os.path.dirname(os.path.abspath(__file__))

DATASET_CARTELLA = os.path.abspath(
     os.path.join(FILE_ATTUALE, "..", "dataset")
)

DATASET_DIR = os.path.abspath(
     os.path.join(DATASET_CARTELLA, "yolo basket.v2i.yolov11"))
DATA_YAML = str(Path(DATASET_DIR) / "data.yaml")



# ============================================================
# CONFIGURAZIONE TRAINING
# ============================================================

MODEL_NAME = "yolo11x.pt"

IMG_SIZE = 1280

EPOCHS = 100
PATIENCE = 20

# Batch totale sulle due GPU.
# Con YOLO11x parti da 16.
# Se la memoria regge, prova 24.
# Se va in errore memoria, scendi a 12 o 8.
BATCH_SIZE = 16

WORKERS = 8

PROJECT_DIR = "runs_basket"
RUN_NAME = "yolo11x_palla_canestro_dual_gpu_1280"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponibile. Controlla driver NVIDIA, PyTorch e GPU.")

    gpu_count = torch.cuda.device_count()
    print(f"GPU disponibili: {gpu_count}")

    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    if gpu_count >= 2:
        device = "0,1"
    else:
        device = "0"
        print("ATTENZIONE: trovata una sola GPU. Userò solo GPU 0.")

    if not Path(DATA_YAML).exists():
        raise FileNotFoundError(f"data.yaml non trovato: {DATA_YAML}")

    print("\nDataset usato:")
    print(DATA_YAML)

    print("\nConfigurazione:")
    print(f"Modello: {MODEL_NAME}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Batch size totale: {BATCH_SIZE}")
    print(f"Device: {device}")
    print(f"Epoche: {EPOCHS}")
    print(f"Patience: {PATIENCE}")

    model = YOLO(MODEL_NAME)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        device=device,
        workers=WORKERS,

        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,

        # Velocizzazione
        cache="ram",
        amp=True,
        optimizer="auto",
        cos_lr=True,

        # ====================================================
        # AUGMENTATION YOLO DISATTIVATA
        # Perché il dataset Roboflow è già preprocessato
        # e già augmentato.
        # ====================================================

        fliplr=0.0,
        flipud=0.0,

        degrees=0.0,
        shear=0.0,
        perspective=0.0,

        translate=0.0,
        scale=0.0,

        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,

        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,

        # Salvataggi
        save=True,
        save_period=5,
        plots=True,
        val=True
    )

    print("\nTraining completato.")

    best_model_path = Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"
    last_model_path = Path(PROJECT_DIR) / RUN_NAME / "weights" / "last.pt"

    print(f"Best model: {best_model_path}")
    print(f"Last model: {last_model_path}")

    evaluate_best_model(best_model_path, device)


def evaluate_best_model(best_model_path, device):
    if not best_model_path.exists():
        print("\nbest.pt non trovato. Salto valutazione finale.")
        return

    print("\nValutazione finale su test set...")

    model = YOLO(str(best_model_path))

    metrics = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=IMG_SIZE,
        device=device,
        verbose=True
    )

    print("\nMetriche per classe:")

    names = metrics.names

    for i, cls_idx in enumerate(metrics.box.ap_class_index):
        cls_idx = int(cls_idx)
        nome_classe = names[cls_idx]

        precision = metrics.box.p[i]
        recall = metrics.box.r[i]
        map50 = metrics.box.ap50[i]
        map5095 = metrics.box.ap[i]

        print(f"\nClasse: {nome_classe}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  mAP50:     {map50:.4f}")
        print(f"  mAP50-95:  {map5095:.4f}")


if __name__ == "__main__":
    main()