"""
swin_unetr_monai.py
Modèle Swin UNETR de MONAI pour la segmentation 3D
"""

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class SwinUNETR3D(nn.Module):
    """
    Wrapper pour Swin UNETR de MONAI
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=3,
        patch_size=2,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        use_checkpoint=False
    ):
        super(SwinUNETR3D, self).__init__()

        self.model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            feature_size=feature_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            use_checkpoint=use_checkpoint
        )

    def forward(self, x):
        """
        x: (B, 1, Z, Y, X)
        Retourne: (B, 3, Z, Y, X) - logits
        """
        return self.model(x)


def test_swin_unetr():
    """Fonction de test"""
    x = torch.randn(2, 1, 64, 64, 64)
    model = SwinUNETR3D(in_channels=1, out_channels=3, feature_size=48)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Paramètres Swin UNETR: {total_params:,}")

    y = model(x)
    print(f"Entrée shape: {x.shape}")
    print(f"Sortie shape: {y.shape}")

    return model


if __name__ == "__main__":
    test_swin_unetr()
