"""
metrics_swin_2class.py
Métriques pour la segmentation binaire (2 classes) - Swin UNETR
"""

import torch
from monai.metrics import DiceMetric, MeanIoU, ConfusionMatrixMetric


class SegmentationMetricsSwin2Class:
    """
    Calcule toutes les métriques pour 2 classes
    """

    def __init__(self, include_background=True, num_classes=2):
        self.include_background = include_background
        self.num_classes = num_classes

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
            metric_name=["sensitivity", "specificity", "precision"],
            reduction="mean_batch"
        )

    def __call__(self, y_pred, y_true):
        if y_true.shape[1] == 1:
            y_true = torch.nn.functional.one_hot(y_true.long(), num_classes=self.num_classes)
            y_true = y_true.squeeze(1).permute(0, 4, 1, 2, 3).float()
        else:
            y_true = y_true.permute(0, 4, 1, 2, 3).float()

        y_pred_softmax = torch.softmax(y_pred, dim=1)
        y_pred_bin = (y_pred_softmax > 0.5).float()
        y_pred_class = torch.argmax(y_pred_softmax, dim=1)
        y_true_class = torch.argmax(y_true, dim=1)

        self.dice_metric(y_pred_bin, y_true)
        self.iou_metric(y_pred_bin, y_true)
        self.confusion_metric(y_pred_bin, y_true)

        dice = self.dice_metric.aggregate()
        iou = self.iou_metric.aggregate()

        confusion_result = self.confusion_metric.aggregate()
        sensitivity = confusion_result[0]
        specificity = confusion_result[1]
        precision = confusion_result[2]

        correct = (y_pred_class == y_true_class).float().sum()
        total = y_true_class.numel()
        accuracy = correct / total

        eps = 1e-6
        sens = sensitivity.mean().item() if sensitivity.numel() > 1 else sensitivity.item()
        prec = precision.mean().item() if precision.numel() > 1 else precision.item()
        f1 = 2 * (prec * sens) / (prec + sens + eps)

        self.dice_metric.reset()
        self.iou_metric.reset()
        self.confusion_metric.reset()

        dice_val = dice.mean().item() if dice.numel() > 1 else dice.item()

        return {
            "dice": dice_val,
            "iou": iou.mean().item() if iou.numel() > 1 else iou.item(),
            "sensitivity": sensitivity.mean().item() if sensitivity.numel() > 1 else sensitivity.item(),
            "specificity": specificity.mean().item() if specificity.numel() > 1 else specificity.item(),
            "precision": precision.mean().item() if precision.numel() > 1 else precision.item(),
            "accuracy": accuracy.item(),
            "f1_score": f1
        }
