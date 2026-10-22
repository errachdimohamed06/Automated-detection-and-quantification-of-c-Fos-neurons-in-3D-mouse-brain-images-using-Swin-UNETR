#!/usr/bin/env python
"""
Script d'évaluation du modèle sur les données de test
"""

import sys
from pathlib import Path

project_root = Path("/mnt/x/segmentation_cfos")
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import pandas as pd

from src.data.dataset_simple import create_datasets_simple
from src.models.unet_3d_monai import UNet3D_MONAI
from src.training.metrics import SegmentationMetrics
from src.inference.predict import load_model, predict_and_save


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 4
    NUM_WORKERS = 0
    
    INIT_FEATURES = 32  
    
    print("=" * 60)
    print("ÉVALUATION DU MODÈLE")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"init_features: {INIT_FEATURES}")
    print("=" * 60)
    print()
    
    data_dir = project_root / "data" / "raw" / "patches_64" / "useful_patches"
    splits_dir = project_root / "data" / "raw" / "patches_64" / "splits"
    
    train_loader, val_loader, test_loader = create_datasets_simple(
        data_dir=data_dir,
        splits_dir=splits_dir,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    
    print(f"Nombre de patches de test: {len(test_loader.dataset)}")
    print()
    
    model_path = project_root / "outputs" / "checkpoints" / "best_model.pth"
    
    if not model_path.exists():
        print(f"❌ Modèle non trouvé: {model_path}")
        print("   Veuillez d'abord entraîner le modèle.")
        return
    
    model = load_model(model_path, UNet3D_MONAI, DEVICE, init_features=INIT_FEATURES)
    print()
    
    print("=" * 60)
    print("CALCUL DES MÉTRIQUES SUR LE TEST")
    print("=" * 60)
    
    metrics_calculator = SegmentationMetrics(include_background=True)
    
    total_metrics = {
        "dice": 0,
        "iou": 0,
        "sensitivity": 0,
        "specificity": 0,
        "precision": 0
    }
    num_batches = 0
    
    model.eval()
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            predictions = model(images)
            predictions = torch.sigmoid(predictions)
            
            metrics = metrics_calculator(predictions, masks)
            
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            
            num_batches += 1
    
    for key in total_metrics:
        total_metrics[key] /= num_batches
    
    print("\nRésultats sur le test set:")
    print("-" * 40)
    for key, value in total_metrics.items():
        print(f"  {key.capitalize()}: {value:.4f}")
    print("-" * 40)
    
    output_dir = project_root / "outputs" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame([total_metrics])
    df.to_csv(output_dir / "test_metrics.csv", index=False)
    print(f"\n✅ Métriques sauvegardées dans {output_dir / 'test_metrics.csv'}")
    
    print("\n" + "=" * 60)
    print("SAUVEGARDE DES PRÉDICTIONS EXEMPLES")
    print("=" * 60)
    
    predict_and_save(
        model=model,
        dataloader=test_loader,
        device=DEVICE,
        output_dir=output_dir / "predictions",
        num_samples=20
    )
    
    print("\n" + "=" * 60)
    print("ÉVALUATION TERMINÉE")
    print("=" * 60)


if __name__ == "__main__":
    main()