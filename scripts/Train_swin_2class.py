#!/usr/bin/env python
"""
Script d'entraînement Swin UNETR - 2 CLASSES
Dataset: pachs_64_2classes_slices40
Utilise les fichiers dédiés pour Swin UNETR
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

from src.data.dataset_swin_2class import create_datasets_swin_2class
from src.models.swin_unetr_monai import SwinUNETR3D
from src.training import get_loss_function
from src.training.metrics_swin_2class import SegmentationMetricsSwin2Class
from src.training.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from src.training.trainer_swin_2class import TrainerSwin2Class


def main():
    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    NUM_WORKERS = 0
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLASSES = 2

    data_dir = Path("/mnt/x/donnees/pachs_64_2classes_slices40")
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ENTRAÎNEMENT SWIN UNETR - 2 CLASSES")
    print("Dataset: pachs_64_2classes_slices40")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Max epochs: {NUM_EPOCHS}")
    print(f"Nombre de classes: {NUM_CLASSES}")
    print("=" * 60)
    print()

    print("Création des datasets (Swin UNETR)...")
    train_loader, val_loader, test_loader = create_datasets_swin_2class(
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

    output_dir = project_root / "outputs" / "swin_2class_slices40"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dossier de sortie: {output_dir}")
    print()

    fieldnames = [
        "epoch",
        "train_loss", "train_dice", "train_iou",
        "train_sensitivity", "train_specificity", "train_precision",
        "train_accuracy", "train_f1_score",
        "val_loss", "val_dice", "val_iou",
        "val_sensitivity", "val_specificity", "val_precision",
        "val_accuracy", "val_f1_score",
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

    resume_checkpoint = checkpoints_dir / "resume_checkpoint.pth"

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

    trainer = TrainerSwin2Class(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        num_classes=NUM_CLASSES,
        callbacks=callbacks,
        checkpoint_path=resume_checkpoint if resume_checkpoint.exists() else None
    )

    start_time = time.time()
    history = trainer.train(num_epochs=NUM_EPOCHS)
    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("SAUVEGARDE FINALE (SWIN UNETR)")
    print("=" * 60)

    df = pd.DataFrame(history)
    df.to_csv(logs_dir / "training_history.csv", index=False)

    config = {
        "experiment": "SwinUNETR_2class_slices40",
        "model": "SwinUNETR",
        "num_classes": NUM_CLASSES,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "total_params": total_params,
        "loss_function": "Dice Loss",
        "data_filtering": "slices_40_2class",
        "total_patches": len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset),
        "best_dice": max(history["val_dice"]),
        "best_f1": max(history["val_f1_score"]),
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
    print("ENTRAÎNEMENT SWIN UNETR - 2 CLASSES TERMINÉ!")
    print("=" * 60)
    print(f"Temps total: {total_time/60:.2f} minutes")
    print(f"Meilleur Dice: {max(history['val_dice']):.4f}")
    print(f"Meilleur F1: {max(history['val_f1_score']):.4f}")


if __name__ == "__main__":
    main()
