"""
src/training/__init__.py
Exporte toutes les fonctions de perte
"""

# Import direct des classes de perte
from src.training.dice_loss import DiceLoss
from src.training.tversky_loss import TverskyLoss
from src.training.focal_loss import FocalLoss
from src.training.combined_loss import CombinedLoss


def get_loss_function(loss_name="dice", **kwargs):
    """
    Fonction utilitaire pour obtenir la fonction de perte par son nom
    
    Paramètres:
        loss_name: "dice", "tversky", "focal", ou "combined"
        **kwargs: paramètres supplémentaires pour la perte
    
    Retourne:
        La fonction de perte demandée
    """
    if loss_name == "dice":
        return DiceLoss(**kwargs)
    elif loss_name == "tversky":
        return TverskyLoss(**kwargs)
    elif loss_name == "focal":
        return FocalLoss(**kwargs)
    elif loss_name == "combined":
        return CombinedLoss(**kwargs)
    else:
        raise ValueError(f"Perte inconnue: {loss_name}. Choisir parmi 'dice', 'tversky', 'focal', 'combined'")


# Liste des éléments exportés
__all__ = [
    "DiceLoss",
    "TverskyLoss", 
    "FocalLoss",
    "CombinedLoss",
    "get_loss_function"
]