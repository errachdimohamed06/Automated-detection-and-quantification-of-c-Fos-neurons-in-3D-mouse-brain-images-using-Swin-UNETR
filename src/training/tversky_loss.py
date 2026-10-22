"""
tversky_loss.py
Tversky Loss pour segmentation multi-classes (3 classes)
"""

import torch
import torch.nn as nn
from monai.losses import TverskyLoss as MonaiTverskyLoss


class TverskyLoss(nn.Module):
    """
    Tversky Loss multi-classes
    """

    def __init__(self, alpha=0.3, beta=0.7, include_background=True, batch=True):
        super(TverskyLoss, self).__init__()
        self.loss = MonaiTverskyLoss(
            alpha=alpha,
            beta=beta,
            include_background=include_background,
            to_onehot_y=True,
            sigmoid=False,
            batch=batch
        )

    def forward(self, y_pred, y_true):
        return self.loss(y_pred, y_true)
