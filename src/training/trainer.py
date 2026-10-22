"""
trainer.py
Boucle d'entraînement et de validation pour U-Net 3D
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import time

from .metrics import compute_all_metrics


class Trainer:
    """
    Classe pour gérer l'entraînement et la validation
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        callbacks=None
    ):
        """
        Paramètres:
            model: Le modèle à entraîner
            train_loader: DataLoader pour l'entraînement
            val_loader: DataLoader pour la validation
            criterion: Fonction de perte
            optimizer: Optimiseur (Adam, AdamW, etc.)
            device: "cuda" ou "cpu"
            callbacks: Dictionnaire contenant les callbacks
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.callbacks = callbacks or {}
        
        # Déplacer le modèle sur le bon device
        self.model = self.model.to(device)
        
        # Historique des métriques
        self.history = {
            "train_loss": [],
            "train_dice": [],
            "val_loss": [],
            "val_dice": [],
            "val_iou": [],
            "val_sensitivity": [],
            "val_precision": [],
            "val_specificity": []
        }
    
    def train_one_epoch(self):
        """
        Entraîne le modèle pendant une époque
        Retourne la perte moyenne et le dice moyen
        """
        self.model.train()  # Mode entraînement (active dropout, batch norm)
        
        total_loss = 0
        total_dice = 0
        num_batches = 0
        
        # Barre de progression
        pbar = tqdm(self.train_loader, desc="Entraînement")
        
        for images, masks in pbar:
            # Déplacer sur GPU
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Réinitialiser les gradients
            self.optimizer.zero_grad()
            
            # Forward pass (prédiction)
            predictions = self.model(images)
            
            # Calcul de la perte
            loss = self.criterion(predictions, masks)
            
            # Backward pass (calcul des gradients)
            loss.backward()
            
            # Mise à jour des poids
            self.optimizer.step()
            
            # Calcul du dice pour monitoring
            dice = compute_all_metrics(predictions, masks)["dice"]
            
            # Accumuler les statistiques
            total_loss += loss.item()
            total_dice += dice
            num_batches += 1
            
            # Mettre à jour la barre de progression
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "dice": f"{dice:.4f}"
            })
        
        # Calculer les moyennes
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        
        return avg_loss, avg_dice
    
    def validate(self):
        """
        Valide le modèle sur les données de validation
        Retourne la perte moyenne et un dictionnaire de métriques
        """
        self.model.eval()  # Mode évaluation (désactive dropout)
        
        total_loss = 0
        total_metrics = {
            "dice": 0,
            "iou": 0,
            "sensitivity": 0,
            "precision": 0,
            "specificity": 0
        }
        num_batches = 0
        
        # Pas de calcul de gradients (économie de mémoire)
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                predictions = self.model(images)
                
                # Calcul de la perte
                loss = self.criterion(predictions, masks)
                
                # Calcul des métriques
                metrics = compute_all_metrics(predictions, masks)
                
                # Accumuler
                total_loss += loss.item()
                for key in total_metrics:
                    total_metrics[key] += metrics[key]
                num_batches += 1
        
        # Calculer les moyennes
        avg_loss = total_loss / num_batches
        for key in total_metrics:
            total_metrics[key] /= num_batches
        
        return avg_loss, total_metrics
    
    def train(self, num_epochs):
        """
        Boucle d'entraînement principale
        
        Paramètres:
            num_epochs: Nombre d'époques maximum
        """
        print("=" * 60)
        print("DÉBUT DE L'ENTRAÎNEMENT")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Nombre d'époques: {num_epochs}")
        print(f"Nombre de batches (train): {len(self.train_loader)}")
        print(f"Nombre de batches (val): {len(self.val_loader)}")
        print("=" * 60)
        print()
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n📊 Époque {epoch}/{num_epochs}")
            print("-" * 40)
            
            # Entraînement
            train_loss, train_dice = self.train_one_epoch()
            
            # Validation
            val_loss, val_metrics = self.validate()
            
            # Sauvegarder l'historique
            self.history["train_loss"].append(train_loss)
            self.history["train_dice"].append(train_dice)
            self.history["val_loss"].append(val_loss)
            self.history["val_dice"].append(val_metrics["dice"])
            self.history["val_iou"].append(val_metrics["iou"])
            self.history["val_sensitivity"].append(val_metrics["sensitivity"])
            self.history["val_precision"].append(val_metrics["precision"])
            self.history["val_specificity"].append(val_metrics["specificity"])
            
            # Afficher les résultats
            print(f"  Train - Loss: {train_loss:.6f}, Dice: {train_dice:.6f}")
            print(f"  Val   - Loss: {val_loss:.6f}, Dice: {val_metrics['dice']:.6f}")
            print(f"          IoU: {val_metrics['iou']:.6f}, Sens: {val_metrics['sensitivity']:.6f}")
            print(f"          Prec: {val_metrics['precision']:.6f}, Spec: {val_metrics['specificity']:.6f}")
            
            # Exécuter les callbacks
            early_stop = False
            
            # Model Checkpoint (sur la loss ou le dice)
            if "model_checkpoint" in self.callbacks:
                monitor_value = val_loss
                if self.callbacks["model_checkpoint"].monitor == "val_dice":
                    monitor_value = val_metrics["dice"]
                
                self.callbacks["model_checkpoint"](
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    current_score=monitor_value
                )
            
            # CSV Logger
            if "csv_logger" in self.callbacks:
                row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_dice": train_dice,
                    "val_loss": val_loss,
                    "val_dice": val_metrics["dice"],
                    "val_iou": val_metrics["iou"],
                    "val_sensitivity": val_metrics["sensitivity"],
                    "val_precision": val_metrics["precision"],
                    "val_specificity": val_metrics["specificity"]
                }
                self.callbacks["csv_logger"].add_row(row)
            
            # Early Stopping (sur la loss ou le dice)
            if "early_stopping" in self.callbacks:
                monitor_value = val_loss
                if self.callbacks["early_stopping"].mode == "max":
                    monitor_value = val_metrics["dice"]
                
                early_stop = self.callbacks["early_stopping"](monitor_value)
                if early_stop:
                    print(f"\n🛑 Early stopping activé après {epoch} époques")
                    break
        
        # Fin de l'entraînement
        total_time = time.time() - start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        
        print("\n" + "=" * 60)
        print("ENTRAÎNEMENT TERMINÉ")
        print("=" * 60)
        print(f"Temps total: {minutes}m {seconds}s")
        print(f"Meilleur validation Dice: {max(self.history['val_dice']):.6f}")
        print("=" * 60)
        
        return self.history