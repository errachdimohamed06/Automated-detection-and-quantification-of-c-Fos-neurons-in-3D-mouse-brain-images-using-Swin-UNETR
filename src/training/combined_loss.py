"""
combined_loss.py
Combinaison Dice + Focal pour segmentation multi-classes
"""

import torch
import torch.nn as nn
from monai.losses import DiceLoss as MonaiDiceLoss
from monai.losses import FocalLoss as MonaiFocalLoss


class CombinedLoss(nn.Module):
    """
    Combined Loss: Dice + Focal
    """

    def __init__(self, dice_weight=0.5, focal_weight=0.5, focal_gamma=2.0):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        self.dice_loss = MonaiDiceLoss(
            include_background=True,
            to_onehot_y=True,
            sigmoid=False,
            batch=True
        )

        self.focal_loss = MonaiFocalLoss(
            gamma=focal_gamma,
            include_background=True,
            to_onehot_y=True,
            sigmoid=False
        )

    def forward(self, y_pred, y_true):
        dice = self.dice_loss(y_pred, y_true)
        focal = self.focal_loss(y_pred, y_true)
        return self.dice_weight * dice + self.focal_weight * focal
