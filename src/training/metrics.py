"""
metrics.py
Métriques pour la segmentation binaire utilisant MONAI
"""

import torch
from monai.metrics import DiceMetric, MeanIoU, ConfusionMatrixMetric


class SegmentationMetrics:
    """
    Classe pour calculer toutes les métriques de segmentation en une fois
    Utilise les implémentations officielles de MONAI
    """
    
    def __init__(self, include_background=True, num_classes=2):
        """
        Paramètres:
            include_background: Inclure la classe fond dans le calcul
            num_classes: Nombre de classes (2 pour binaire: fond et neurones)
        """
        self.include_background = include_background
        self.num_classes = num_classes
        
        # Initialiser les métriques MONAI
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
        
        # Pour la sensibilité, spécificité, précision
        self.confusion_metric = ConfusionMatrixMetric(
            include_background=include_background,
            metric_name=["sensitivity", "specificity", "precision"],
            reduction="mean_batch"
        )
    
    def __call__(self, y_pred, y_true):
        """
        Calcule toutes les métriques
        
        y_pred: Prédiction (logits ou probabilités)
        y_true: Mask réel (0 et 1)
        
        Retourne: dictionnaire avec toutes les métriques
        """
        # Binariser les prédictions (seuil à 0.5)
        y_pred_bin = (y_pred > 0.5).float()
        
        # Mettre à jour et calculer les métriques
        self.dice_metric(y_pred_bin, y_true)
        self.iou_metric(y_pred_bin, y_true)
        self.confusion_metric(y_pred_bin, y_true)
        
        # Récupérer les valeurs
        dice = self.dice_metric.aggregate()
        iou = self.iou_metric.aggregate()
        
        # Pour la confusion matrix, le résultat est un tuple (métriques, not_nans)
        confusion_result = self.confusion_metric.aggregate()
        if isinstance(confusion_result, tuple):
            sensitivity, specificity, precision = confusion_result[0]
        else:
            sensitivity, specificity, precision = confusion_result
        
        # Remettre à zéro pour le prochain batch
        self.dice_metric.reset()
        self.iou_metric.reset()
        self.confusion_metric.reset()
        
        # Retourner les métriques sous forme de dictionnaire
        return {
            "dice": dice.item() if dice.numel() == 1 else dice.mean().item(),
            "iou": iou.item() if iou.numel() == 1 else iou.mean().item(),
            "sensitivity": sensitivity.item() if sensitivity.numel() == 1 else sensitivity.mean().item(),
            "specificity": specificity.item() if specificity.numel() == 1 else specificity.mean().item(),
            "precision": precision.item() if precision.numel() == 1 else precision.mean().item()
        }


def compute_all_metrics(y_pred, y_true):
    """
    Fonction simple pour calculer toutes les métriques en une fois
    
    y_pred: Prédiction (logits ou probabilités)
    y_true: Mask réel (0 et 1)
    """
    metrics = SegmentationMetrics()
    return metrics(y_pred, y_true)


# Fonctions individuelles si on veut les utiliser séparément
def dice_score(y_pred, y_true, include_background=True):
    """Dice Score avec MONAI"""
    y_pred_bin = (y_pred > 0.5).float()
    metric = DiceMetric(include_background=include_background, reduction="mean_batch")
    metric(y_pred_bin, y_true)
    result = metric.aggregate()
    metric.reset()
    return result.item() if result.numel() == 1 else result.mean().item()


def iou_score(y_pred, y_true, include_background=True):
    """IoU (Intersection over Union) avec MONAI"""
    y_pred_bin = (y_pred > 0.5).float()
    metric = MeanIoU(include_background=include_background, reduction="mean_batch")
    metric(y_pred_bin, y_true)
    result = metric.aggregate()
    metric.reset()
    return result.item() if result.numel() == 1 else result.mean().item()


def sensitivity(y_pred, y_true, include_background=True):
    """Sensibilité (Recall) avec MONAI"""
    y_pred_bin = (y_pred > 0.5).float()
    metric = ConfusionMatrixMetric(
        include_background=include_background,
        metric_name="sensitivity",
        reduction="mean_batch"
    )
    metric(y_pred_bin, y_true)
    result = metric.aggregate()
    metric.reset()
    # La sortie peut être un tuple
    if isinstance(result, tuple):
        result = result[0]
    return result.item() if result.numel() == 1 else result.mean().item()


def precision(y_pred, y_true, include_background=True):
    """Précision avec MONAI"""
    y_pred_bin = (y_pred > 0.5).float()
    metric = ConfusionMatrixMetric(
        include_background=include_background,
        metric_name="precision",
        reduction="mean_batch"
    )
    metric(y_pred_bin, y_true)
    result = metric.aggregate()
    metric.reset()
    if isinstance(result, tuple):
        result = result[0]
    return result.item() if result.numel() == 1 else result.mean().item()


def specificity(y_pred, y_true, include_background=True):
    """Spécificité avec MONAI"""
    y_pred_bin = (y_pred > 0.5).float()
    metric = ConfusionMatrixMetric(
        include_background=include_background,
        metric_name="specificity",
        reduction="mean_batch"
    )
    metric(y_pred_bin, y_true)
    result = metric.aggregate()
    metric.reset()
    if isinstance(result, (list, tuple)):
        result = result[0]

    if not torch.is_tensor(result):
        result = torch.as_tensor(result)
    return result.item() if result.numel() == 1 else result.mean().item()


# Fonction de test
if __name__ == "__main__":
    # Créer des faux batch
    batch_size = 4
    y_pred = torch.rand(batch_size, 1, 64, 64, 64)  # Prédictions aléatoires
    y_true = torch.randint(0, 2, (batch_size, 1, 64, 64, 64)).float()  # Masks aléatoires
    
    # Tester la fonction principale
    metrics = compute_all_metrics(y_pred, y_true)
    print("Métriques calculées avec MONAI:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Tester les fonctions individuelles
    print("\nFonctions individuelles:")
    print(f"  Dice: {dice_score(y_pred, y_true):.4f}")
    print(f"  IoU: {iou_score(y_pred, y_true):.4f}")
    print(f"  Sensitivity: {sensitivity(y_pred, y_true):.4f}")
    print(f"  Precision: {precision(y_pred, y_true):.4f}")
    print(f"  Specificity: {specificity(y_pred, y_true):.4f}")