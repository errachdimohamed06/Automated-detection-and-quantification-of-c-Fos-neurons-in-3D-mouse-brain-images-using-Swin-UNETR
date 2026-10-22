"""
focal_loss.py
Focal Loss pour segmentation multi-classes (3 classes)
"""

import torch
import torch.nn as nn
from monai.losses import FocalLoss as MonaiFocalLoss


class FocalLoss(nn.Module):
    """
    Focal Loss multi-classes
    """

    def __init__(self, gamma=2.0, include_background=True):
        super(FocalLoss, self).__init__()
        self.loss = MonaiFocalLoss(
            gamma=gamma,
            include_background=include_background,
            to_onehot_y=True
        )

    def forward(self, y_pred, y_true):
        return self.loss(y_pred, y_true)
