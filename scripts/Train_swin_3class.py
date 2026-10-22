#!/usr/bin/env python
"""
Script d'entraînement Swin UNETR - 3 classes
REPRISE DEPUIS best_model.pth avec nouvel historique
"""

import sys
import time
import json
from pathlib import Path

project_root = Path("/mnt/x/segmentation_cfos")
sys.path.insert(0, str(project_root))

import torch
import torch.optim as optim
import pandas as pd

from src.data.dataset_3class import create_datasets_3class
from src.models.swin_unetr_monai import SwinUNETR3D
from src.training import get_loss_function
from src.training.metrics_3class import SegmentationMetrics3Class
from src.training.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from src.training.trainer_3class import Trainer3Class


def main():
    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    NUM_WORKERS = 0
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLASSES = 3

    data_dir = Path("/mnt/x/donnees/pachs_64_3classes_filtered_slices")
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ENTRAÎNEMENT SWIN UNETR - 3 CLASSES (REPRISE BEST_MODEL)")
    print("FILTRAGE: SEUIL 40 SLICES")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Max epochs: {NUM_EPOCHS}")
    print(f"Nombre de classes: {NUM_CLASSES}")
    print("=" * 60)
    print()

    print("Création des datasets...")
    train_loader, val_loader, test_loader = create_datasets_3class(
        data_dir=data_dir,
        splits_dir=splits_dir,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    print(f"Train: {len(train_loader.dataset)} patches")
    print(f"Val: {len(val_loader.dataset)} patches")
    print(f"Test: {len(test_loader.dataset)} patches")
    print()

    model = SwinUNETR3D(
        in_channels=1,
        out_channels=NUM_CLASSES,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        use_checkpoint=False
    )

    best_model_path = project_root / "outputs" / "swin_3class_slices40_focal" / "checkpoints" / "best_model.pth"

    if best_model_path.exists():
        print(f"🔄 Chargement du meilleur modèle: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_score = checkpoint.get("best_score", 0.6244)
        start_epoch = 13
        print(f"   Meilleur score: {best_score:.6f}")
        print(f"   Reprise à l'époque: {start_epoch}")
    else:
        print(f"❌ Modèle non trouvé: {best_model_path}")
        return

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Modèle: Swin UNETR")
    print(f"Paramètres: {total_params:,}")
    print()

    criterion = get_loss_function("focal", gamma=2.0)
    print(f"Loss: Focal Loss (gamma=2.0)")
    print()

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print(f"Optimiseur: AdamW")
    print()

    output_dir = project_root / "outputs" / "swin_3class_slices40_focal_resume"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Nouveau dossier de sortie: {output_dir}")
    print()

    fieldnames = [
        "epoch",
        "train_loss", "train_dice", "train_iou",
        "val_loss", "val_dice", "val_iou",
        "val_sensitivity", "val_specificity", "val_precision",
        "val_accuracy", "val_f1_score",
        "val_dice_class0", "val_dice_class1", "val_dice_class2",
        "epoch_time_seconds", "train_time_seconds", "val_time_seconds",
        "gpu_memory_mb"
    ]

    csv_logger = CSVLogger(
        filename=logs_dir / "training_log.csv",
        fieldnames=fieldnames
    )

    checkpoint_best = ModelCheckpoint(
        filepath=checkpoints_dir / "best_model.pth",
        monitor="val_dice",
        mode="max",
        save_best_only=True
    )

    checkpoint_last = ModelCheckpoint(
        filepath=checkpoints_dir / "last_model.pth",
        monitor="epoch",
        mode="max",
        save_best_only=False
    )

    early_stopping = EarlyStopping(
        patience=15,
        min_delta=0.001,
        mode="max"
    )

    callbacks = {
        "csv_logger": csv_logger,
        "model_checkpoint": checkpoint_best,
        "model_checkpoint_last": checkpoint_last,
        "early_stopping": early_stopping
    }

    trainer = Trainer3Class(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        num_classes=NUM_CLASSES,
        callbacks=callbacks,
        checkpoint_path=None  
    )

    trainer.start_epoch = start_epoch

    start_time = time.time()
    history = trainer.train(num_epochs=NUM_EPOCHS)
    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("SAUVEGARDE FINALE")
    print("=" * 60)

    df = pd.DataFrame(history)
    df.to_csv(logs_dir / "training_history.csv", index=False)

    config = {
        "experiment": "SwinUNETR_3class_slices40_focal_resume",
        "model": "SwinUNETR",
        "num_classes": NUM_CLASSES,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "total_params": total_params,
        "loss_function": "Focal Loss",
        "gamma": 2.0,
        "data_filtering": "slices_40",
        "resume_from_best": str(best_model_path),
        "total_patches": len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset),
        "best_dice": max(history["val_dice"]) if len(history["val_dice"]) > 0 else 0,
        "best_f1": max(history["val_f1_score"]) if len(history["val_f1_score"]) > 0 else 0,
        "total_time_minutes": total_time / 60,
        "time_per_epoch_minutes": (total_time / 60) / len(history["epoch"]) if len(history["epoch"]) > 0 else 0,
        "device": DEVICE
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"✅ Historique sauvegardé: {logs_dir / 'training_history.csv'}")
    print(f"✅ Configuration sauvegardée: {output_dir / 'config.json'}")
    print(f"✅ Meilleur modèle: {checkpoints_dir / 'best_model.pth'}")
    print(f"✅ Dernier modèle: {checkpoints_dir / 'last_model.pth'}")
    print()
    print("=" * 60)
    print("ENTRAÎNEMENT SWIN UNETR TERMINÉ!")
    print("=" * 60)
    print(f"Temps total: {total_time/60:.2f} minutes")
    print(f"Meilleur Dice: {max(history['val_dice']):.4f}" if len(history['val_dice']) > 0 else "Aucune métrique")
    print(f"Meilleur F1: {max(history['val_f1_score']):.4f}" if len(history['val_f1_score']) > 0 else "Aucune métrique")


if __name__ == "__main__":
    main()
