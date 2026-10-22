#!/usr/bin/env python
"""
Script d'entraînement U-NET 3D - 2 CLASSES
Patches filtrés par slices (seuil 40)
Conversion 3 classes → 2 classes (vaisseaux → neurones)
Toutes les métriques : Accuracy, F1-Score, Recall, Precision, Dice, IoU
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
import numpy as np

from src.data.dataset_2class import create_datasets_2class
from src.models.unet_3d_monai import UNet3D_MONAI
from src.training import get_loss_function
from src.training.metrics_2class import SegmentationMetrics2Class
from src.training.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from src.training.trainer_2class import Trainer2Class


def main():
    BATCH_SIZE = 16  
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    NUM_WORKERS = 0
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLASSES = 2  
    
    data_dir = Path("/mnt/x/donnees/pachs_64_2classes_slices40")
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    print("ENTRAÎNEMENT U-NET 3D - 2 CLASSES")
    print("CLASSE 1: NEURONES + VAISSEAUX")
    print("FILTRAGE: SEUIL 40 SLICES")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Max epochs: {NUM_EPOCHS}")
    print(f"Nombre de classes: {NUM_CLASSES}")
    
    print("Création des datasets...")
    train_loader, val_loader, test_loader = create_datasets_2class(
        data_dir=data_dir,
        splits_dir=splits_dir,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    
    print(f"Train: {len(train_loader.dataset)} patches")
    print(f"Val: {len(val_loader.dataset)} patches")
    print(f"Test: {len(test_loader.dataset)} patches")
    print()
    
    model = UNet3D_MONAI(
        in_channels=1,
        out_channels=NUM_CLASSES,
        init_features=32
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Modèle: U-Net 3D")
    print(f"Paramètres: {total_params:,}")
    print()
    
    criterion = get_loss_function("dice")
    print(f"Loss: Dice Loss (binaire)")
    print()
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print(f"Optimiseur: AdamW")
    print()
    
    output_dir = project_root / "outputs" / "unet_2class_slices40"
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
        "train_accuracy", "train_f1", "train_recall", "train_precision",
        "val_loss", "val_dice", "val_iou",
        "val_accuracy", "val_f1", "val_recall", "val_precision",
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
    
    trainer = Trainer2Class(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        callbacks=callbacks
    )
    
    start_time = time.time()
    history = trainer.train(num_epochs=NUM_EPOCHS)
    total_time = time.time() - start_time  
    print("SAUVEGARDE FINALE")    
    df = pd.DataFrame(history)
    df.to_csv(logs_dir / "training_history.csv", index=False)
    
    config = {
        "experiment": "UNet3D_2class_slices40_dice",
        "model": "UNet3D_MONAI",
        "num_classes": NUM_CLASSES,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "total_params": total_params,
        "loss_function": "Dice Loss",
        "data_filtering": "slices_40",
        "data_conversion": "3classes_to_2classes",
        "total_patches": len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset),
        "best_dice": max(history["val_dice"]),
        "best_iou": max(history["val_iou"]),
        "best_f1": max(history["val_f1"]),
        "best_accuracy": max(history["val_accuracy"]),
        "total_time_minutes": total_time / 60,
        "time_per_epoch_minutes": (total_time / 60) / len(history["epoch"]),
        "device": DEVICE
    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Historique sauvegardé: {logs_dir / 'training_history.csv'}")
    print(f"✅ Configuration sauvegardée: {output_dir / 'config.json'}")
    print(f"✅ Meilleur modèle: {checkpoints_dir / 'best_model.pth'}")
    print(f"✅ Dernier modèle: {checkpoints_dir / 'last_model.pth'}")
    print("ENTRAÎNEMENT U-NET 3D (2 CLASSES) TERMINÉ!")
    print(f"Temps total: {total_time/60:.2f} minutes")
    print(f"Meilleur Dice: {max(history['val_dice']):.4f}")
    print(f"Meilleur F1: {max(history['val_f1']):.4f}")
    print(f"Meilleure Accuracy: {max(history['val_accuracy']):.4f}")


if __name__ == "__main__":
    main()