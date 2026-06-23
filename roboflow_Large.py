import torch
from pathlib import Path
from rfdetr import RFDETRLarge


DATASET_DIR = "/home/vrlab/Scrivania/BasketAR/Gruppo19/dataset/dataset_rfdetr"
OUTPUT_DIR = "/home/vrlab/Scrivania/BasketAR/Gruppo19/dataset/runs_rfdetr_896/rfdetr_large_palla_canestro_896"


def main():
    dataset_path = Path(DATASET_DIR)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset non trovato: {dataset_path}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponibile. Controlla driver NVIDIA e PyTorch CUDA.")

    print("GPU disponibile:", torch.cuda.get_device_name(0))

    model = RFDETRLarge()

    model.train(
        dataset_dir=str(dataset_path),
        output_dir=OUTPUT_DIR,

        # Training
        epochs=100,
        batch_size=2,
        grad_accum_steps=8,

        # Learning rate
        lr=1e-4,
        lr_encoder=1e-5,

        # Risoluzione vicina a 880, ma più sicura
        resolution=896,

        # GPU
        device="cuda",

        # Salvataggi
        checkpoint_interval=5,

        # Early stopping
        early_stopping=True,
        early_stopping_patience=20,

        # Utile se la memoria GPU non basta
        gradient_checkpointing=True,

        # EMA: spesso migliora la generalizzazione
        use_ema=True,

        # No Weights & Biases
        wandb=False
    )


if __name__ == "__main__":
    main()