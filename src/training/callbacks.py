"""
callbacks.py
Callbacks pour l'entraînement
"""

import os
import csv
import torch


class EarlyStopping:
    """Arrête l'entraînement quand la métrique ne s'améliore plus"""
    
    def __init__(self, patience=10, min_delta=0.001, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return False
        
        if self.mode == "min":
            improved = current_score < self.best_score - self.min_delta
        else:
            improved = current_score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class ModelCheckpoint:
    """Sauvegarde le meilleur modèle"""
    
    def __init__(self, filepath, monitor="val_loss", mode="min", save_best_only=True):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = None
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    def __call__(self, epoch, model, optimizer, current_score):
        if self.best_score is None:
            self.best_score = current_score
            self._save(epoch, model, optimizer, current_score)
            return True
        
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
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_score": current_score
        }
        torch.save(checkpoint, self.filepath)
        print(f"  💾 Modèle sauvegardé: {self.filepath} (score: {current_score:.6f})")


class CSVLogger:
    """Sauvegarde les métriques dans un fichier CSV"""
    
    def __init__(self, filename, fieldnames):
        self.filename = filename
        self.fieldnames = fieldnames
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
    def add_row(self, row_data):
        with open(self.filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row_data)