"""
callbacks.py
Callbacks pour l'entraînement
- Early Stopping: arrête l'entraînement si la performance ne s'améliore plus
- Model Checkpoint: sauvegarde le meilleur modèle
- CSV Logger: sauvegarde les métriques dans un fichier CSV
"""

import os
import csv
import torch


class EarlyStopping:
    """
    Early Stopping
    Arrête l'entraînement quand la métrique de validation ne s'améliore plus
    """
    
    def __init__(self, patience=10, min_delta=0.001, mode="min"):
        """
        Paramètres:
            patience: Nombre d'époques à attendre sans amélioration
            min_delta: Amélioration minimale pour considérer comme meilleur
            mode: "min" pour loss, "max" pour dice (la métrique à surveiller)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, current_score):
        """
        Vérifie si on doit arrêter l'entraînement
        Retourne True si on doit arrêter
        """
        if self.best_score is None:
            # Première époque: on sauvegarde le score
            self.best_score = current_score
            return False
        
        # Vérifier si le score s'est amélioré
        if self.mode == "min":
            # Pour la loss: plus bas = meilleur
            improved = current_score < self.best_score - self.min_delta
        else:
            # Pour dice: plus haut = meilleur
            improved = current_score > self.best_score + self.min_delta
        
        if improved:
            # Amélioration: on reset le compteur
            self.best_score = current_score
            self.counter = 0
        else:
            # Pas d'amélioration: on incrémente le compteur
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class ModelCheckpoint:
    """
    Model Checkpoint
    Sauvegarde le meilleur modèle basé sur une métrique
    """
    
    def __init__(self, filepath, monitor="val_loss", mode="min", save_best_only=True):
        """
        Paramètres:
            filepath: Chemin pour sauvegarder le modèle
            monitor: Métrique à surveiller ("val_loss" ou "val_dice")
            mode: "min" pour loss, "max" pour dice
            save_best_only: Sauvegarde seulement si c'est le meilleur
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = None
        
        # Créer le dossier parent s'il n'existe pas
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    def __call__(self, epoch, model, optimizer, current_score):
        """
        Sauvegarde le modèle si nécessaire
        """
        # Première sauvegarde ou vérification d'amélioration
        if self.best_score is None:
            self.best_score = current_score
            self._save(epoch, model, optimizer, current_score)
            return True
        
        # Vérifier si le score s'est amélioré
        if self.mode == "min":
            improved = current_score < self.best_score
        else:
            improved = current_score > self.best_score
        
        if improved:
            self.best_score = current_score
            if self.save_best_only:
                self._save(epoch, model, optimizer, current_score)
            return True
        
        return False
    
    def _save(self, epoch, model, optimizer, current_score):
        """
        Sauvegarde le modèle
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_score": current_score
        }
        torch.save(checkpoint, self.filepath)
        print(f"  💾 Modèle sauvegardé: {self.filepath} (score: {current_score:.6f})")


class CSVLogger:
    """
    CSV Logger
    Sauvegarde toutes les métriques dans un fichier CSV
    """
    
    def __init__(self, filename, fieldnames):
        """
        Paramètres:
            filename: Chemin du fichier CSV
            fieldnames: Liste des noms de colonnes
        """
        self.filename = filename
        self.fieldnames = fieldnames
        
        # Créer le dossier parent s'il n'existe pas
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Créer le fichier CSV avec l'en-tête
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
    def add_row(self, row_data):
        """
        Ajoute une ligne au CSV
        """
        with open(self.filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row_data)