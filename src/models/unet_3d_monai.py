"""
unet_3d_monai.py
Modèle U-Net 3D de MONAI pour la segmentation de neurones
"""

import torch
import torch.nn as nn
from monai.networks.nets import UNet


class UNet3D_MONAI(nn.Module):
    """
    Wrapper pour le U-Net 3D de MONAI
    """
    
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet3D_MONAI, self).__init__()
        
        self.model = UNet(
            spatial_dims=3,                    # 3D
            in_channels=in_channels,           # 1 (image en niveaux de gris)
            out_channels=out_channels,         # 1 (mask binaire)
            channels=(init_features, init_features*2, init_features*4, init_features*8, init_features*16),
            strides=(2, 2, 2, 2),              # Réduction de taille à chaque niveau
            num_res_units=2,                   # Nombre de convolutions par bloc
            norm="batch",                      # BatchNorm (ou "instance")
            dropout=0.0                        # Pas de dropout
        )
    
    def forward(self, x):
        return torch.sigmoid(self.model(x))    # Sigmoid pour avoir des valeurs entre 0 et 1


def test_unet3d():
    """Fonction de test"""
    x = torch.randn(4, 1, 64, 64, 64)
    model = UNet3D_MONAI(in_channels=1, out_channels=1, init_features=32)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Nombre de paramètres: {total_params:,}")
    
    y = model(x)
    print(f"Entrée shape: {x.shape}")
    print(f"Sortie shape: {y.shape}")
    print(f"Valeurs min: {y.min().item():.4f}, max: {y.max().item():.4f}")
    
    return model


if __name__ == "__main__":
    test_unet3d()