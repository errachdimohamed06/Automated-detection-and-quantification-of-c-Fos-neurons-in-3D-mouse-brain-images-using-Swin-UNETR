"""
predict.py
Fonctions pour charger un modèle et faire des prédictions sur les patches de test
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm


def load_model(model_path, model_class, device="cuda", init_features=64):
    """
    Charge un modèle sauvegardé
    
    Paramètres:
        model_path: Chemin vers le fichier .pth
        model_class: La classe du modèle (ex: UNet3D_MONAI)
        device: "cuda" ou "cpu"
        init_features: Nombre de filtres initiaux (doit correspondre à l'entraînement)
    
    Retourne:
        Le modèle chargé
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Créer une instance du modèle avec init_features
    model = model_class(in_channels=1, out_channels=1, init_features=init_features)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Modèle chargé depuis: {model_path}")
    print(f"   Époque: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Score: {checkpoint.get('best_score', 'N/A'):.6f}")
    
    return model


def predict_patch(model, patch, device="cuda"):
    """
    Fait une prédiction sur un patch
    """
    if patch.ndim == 3:
        patch = patch[np.newaxis, np.newaxis, ...]
    elif patch.ndim == 4:
        patch = patch[np.newaxis, ...]
    
    patch_tensor = torch.from_numpy(patch).float().to(device)
    
    with torch.no_grad():
        pred = model(patch_tensor)
    
    pred_np = pred.cpu().numpy().squeeze()
    
    return pred_np


def predict_batch(model, patches, device="cuda"):
    """
    Fait des prédictions sur un batch de patches
    """
    if isinstance(patches, list):
        patches = np.stack(patches, axis=0)
    
    if patches.ndim == 4:
        patches = patches[:, np.newaxis, ...]
    
    patch_tensor = torch.from_numpy(patches).float().to(device)
    
    with torch.no_grad():
        preds = model(patch_tensor)
    
    preds_np = preds.cpu().numpy().squeeze(axis=1)
    
    if preds_np.shape[0] == 1:
        return preds_np[0]
    
    return preds_np


def predict_and_save(model, dataloader, device, output_dir, num_samples=10):
    """
    Prédit sur quelques échantillons du dataloader et sauvegarde les résultats
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    samples_saved = 0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Prédiction"):
            images = images.to(device)
            masks = masks.to(device)
            
            predictions = model(images)
            predictions = torch.sigmoid(predictions)
            
            images_np = images.cpu().numpy()
            masks_np = masks.cpu().numpy()
            preds_np = predictions.cpu().numpy()
            
            for i in range(images_np.shape[0]):
                if samples_saved >= num_samples:
                    return
                
                sample_data = {
                    "image": images_np[i, 0, ...],
                    "mask": masks_np[i, 0, ...],
                    "prediction": preds_np[i, 0, ...]
                }
                
                np.savez_compressed(
                    output_dir / f"sample_{samples_saved:04d}.npz",
                    image=sample_data["image"],
                    mask=sample_data["mask"],
                    prediction=sample_data["prediction"]
                )
                
                samples_saved += 1
                print(f"  ✅ Sauvegardé sample_{samples_saved:04d}.npz")
    
    print(f"\n✅ {samples_saved} échantillons sauvegardés dans {output_dir}")