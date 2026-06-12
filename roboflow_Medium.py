from rfdetr import RFDETRMedium
import os
FILE_ATTUALE = os.path.dirname(os.path.abspath(__file__))

DATASET_CARTELLA = os.path.abspath(
     os.path.join(FILE_ATTUALE, "..", "dataset")
)
DATASET_DIR=os.path.join(DATASET_CARTELLA, "yolo basket.v2i.yolov11")
OUTPUT_DIR=os.path.join(DATASET_CARTELLA, "rfdetr_medium")
model = RFDETRMedium()

model.train(
    dataset_dir=str(DATASET_DIR),
    output_dir=str(OUTPUT_DIR),

    epochs=100,
    resolution=576,

    # Valori per singola GPU
    batch_size=4,
    grad_accum_steps=2,

    lr=1e-4,
    lr_encoder=1.5e-4,
    weight_decay=1e-4,

    # Necessario per il multi-GPU RF-DETR
    devices="auto",

    # Il dataset è già augmentato da Roboflow
    aug_config={},

    early_stopping=True,
    early_stopping_patience=15,
    early_stopping_min_delta=0.001,
    early_stopping_use_ema=True,

    checkpoint_interval=5,
    use_ema=True,

    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,

    progress_bar="tqdm",
    seed=42,
)