"""
metrics_2class.py
Métriques pour la segmentation binaire (2 classes)
"""

import torch
from monai.metrics import DiceMetric, MeanIoU, ConfusionMatrixMetric


class SegmentationMetrics2Class:
    """
    Calcule toutes les métriques pour 2 classes
    """
    
    def __init__(self, include_background=True):
        self.include_background = include_background
        
        self.dice_metric = DiceMetric(
            include_background=include_background,
            reduction="mean_batch",
            ignore_empty=True
        )
        
        self.iou_metric = MeanIoU(
            include_background=include_background,
            reduction="mean_batch",
            ignore_empty=True
        )
        
        self.confusion_metric = ConfusionMatrixMetric(
            include_background=include_background,
            metric_name=["sensitivity", "specificity", "precision", "accuracy", "f1 score"],
            reduction="mean_batch"
        )
    
    def __call__(self, y_pred, y_true):
        """
        Calcule toutes les métriques
        
        y_pred: shape (B, 2, Z, Y, X)
        y_true: shape (B, 1, Z, Y, X)
        """
        # Binariser la prédiction
        y_pred_softmax = torch.softmax(y_pred, dim=1)
        y_pred_bin = (y_pred_softmax > 0.5).float()
        
        # Convertir y_true en one-hot
        y_true_onehot = torch.nn.functional.one_hot(y_true.long(), num_classes=2)
        y_true_onehot = y_true_onehot.squeeze(1).permute(0, 4, 1, 2, 3).float()
        
        # Calculer les métriques
        self.dice_metric(y_pred_bin, y_true_onehot)
        self.iou_metric(y_pred_bin, y_true_onehot)
        self.confusion_metric(y_pred_bin, y_true_onehot)
        
        # Récupérer les valeurs
        dice = self.dice_metric.aggregate()
        iou = self.iou_metric.aggregate()
        
        confusion_result = self.confusion_metric.aggregate()
        if isinstance(confusion_result, tuple):
            sensitivity, specificity, precision, accuracy, f1 = confusion_result[0]
        else:
            sensitivity, specificity, precision, accuracy, f1 = confusion_result
        
        self.dice_metric.reset()
        self.iou_metric.reset()
        self.confusion_metric.reset()
        
        return {
            "dice": dice.mean().item() if dice.numel() > 1 else dice.item(),
            "iou": iou.mean().item() if iou.numel() > 1 else iou.item(),
            "sensitivity": sensitivity.mean().item() if sensitivity.numel() > 1 else sensitivity.item(),
            "specificity": specificity.mean().item() if specificity.numel() > 1 else specificity.item(),
            "precision": precision.mean().item() if precision.numel() > 1 else precision.item(),
            "accuracy": accuracy.mean().item() if accuracy.numel() > 1 else accuracy.item(),
            "f1": f1.mean().item() if f1.numel() > 1 else f1.item()
        }