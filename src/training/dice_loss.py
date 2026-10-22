"""
dice_loss.py
Dice Loss pour segmentation multi-classes (3 classes)
"""

import torch
import torch.nn as nn
from monai.losses import DiceLoss as MonaiDiceLoss


class DiceLoss(nn.Module):
    """
    Dice Loss multi-classes
    """

    def __init__(self, include_background=True, sigmoid=False, batch=True):
        super(DiceLoss, self).__init__()
        self.loss = MonaiDiceLoss(
            include_background=include_background,
            to_onehot_y=True,     # Convertit le mask en one-hot (3 canaux)
            sigmoid=sigmoid,      # False car on utilise softmax (multi-classes)
            batch=batch
        )

    def forward(self, y_pred, y_true):
        """
        y_pred: (B, 3, Z, Y, X) - logits (pas de sigmoid/softmax)
        y_true: (B, 1, Z, Y, X) - valeurs 0, 1, 2
        """
        return self.loss(y_pred, y_true)
