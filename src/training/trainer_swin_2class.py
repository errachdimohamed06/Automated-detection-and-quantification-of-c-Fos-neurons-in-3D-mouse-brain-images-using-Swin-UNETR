"""
trainer_swin_2class.py
Boucle d'entraînement pour 2 classes avec reprise - Swin UNETR
"""

import torch
import time
import gc
from tqdm import tqdm
from pathlib import Path
from src.training.metrics_swin_2class import SegmentationMetricsSwin2Class


class TrainerSwin2Class:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer,
                 device, num_classes=2, callbacks=None, checkpoint_path=None):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_classes = num_classes
        self.callbacks = callbacks or {}
        self.checkpoint_path = checkpoint_path

        self.metrics_calc = SegmentationMetricsSwin2Class(include_background=True, num_classes=num_classes)

        self.history = {
            "epoch": [],
            "train_loss": [], "train_dice": [], "train_iou": [],
            "train_sensitivity": [], "train_specificity": [], "train_precision": [],
            "train_accuracy": [], "train_f1_score": [],
            "val_loss": [], "val_dice": [], "val_iou": [],
            "val_sensitivity": [], "val_specificity": [], "val_precision": [],
            "val_accuracy": [], "val_f1_score": [],
            "epoch_time_seconds": [], "train_time_seconds": [], "val_time_seconds": [],
            "gpu_memory_mb": []
        }

        self.start_epoch = 1

        if checkpoint_path and checkpoint_path.exists():
            self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, checkpoint_path):
        print(f"🔄 Reprise depuis: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.history = checkpoint.get("history", self.history)
        print(f"   Reprise à l'époque {self.start_epoch}")

    def get_gpu_memory(self):
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        total_metrics = {
            "dice": 0, "iou": 0, "sensitivity": 0, "specificity": 0,
            "precision": 0, "accuracy": 0, "f1_score": 0
        }
        num_batches = 0
        start_time = time.time()

        pbar = tqdm(self.train_loader, desc="Entraînement (Swin)")
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(images)
            loss = self.criterion(predictions, masks)
            loss.backward()
            self.optimizer.step()

            metrics = self.metrics_calc(predictions, masks)

            total_loss += loss.item()
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "dice": f"{metrics['dice']:.4f}",
                "iou": f"{metrics['iou']:.4f}"
            })

        epoch_time = time.time() - start_time
        avg_loss = total_loss / num_batches
        for key in total_metrics:
            total_metrics[key] /= num_batches

        return avg_loss, total_metrics, epoch_time

    def validate(self):
        self.model.eval()
        total_loss = 0
        total_metrics = {
            "dice": 0, "iou": 0, "sensitivity": 0, "specificity": 0,
            "precision": 0, "accuracy": 0, "f1_score": 0
        }
        num_batches = 0
        start_time = time.time()

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation (Swin)")
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)

                predictions = self.model(images)
                loss = self.criterion(predictions, masks)
                metrics = self.metrics_calc(predictions, masks)

                total_loss += loss.item()
                for key in total_metrics:
                    total_metrics[key] += metrics[key]
                num_batches += 1

        val_time = time.time() - start_time
        avg_loss = total_loss / num_batches
        for key in total_metrics:
            total_metrics[key] /= num_batches

        return avg_loss, total_metrics, val_time

    def train(self, num_epochs):
        print("=" * 60)
        print("DÉBUT DE L'ENTRAÎNEMENT (SWIN UNETR)" + (" - REPRISE" if self.start_epoch > 1 else ""))
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Époque de départ: {self.start_epoch}")
        print("=" * 60)
        print()

        total_start_time = time.time()

        for epoch in range(self.start_epoch, num_epochs + 1):
            print(f"\n📊 Époque {epoch}/{num_epochs} (Swin UNETR)")
            print("-" * 40)

            train_loss, train_metrics, train_time = self.train_one_epoch()
            val_loss, val_metrics, val_time = self.validate()
            gpu_memory = self.get_gpu_memory()

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["train_dice"].append(train_metrics["dice"])
            self.history["train_iou"].append(train_metrics["iou"])
            self.history["train_sensitivity"].append(train_metrics["sensitivity"])
            self.history["train_specificity"].append(train_metrics["specificity"])
            self.history["train_precision"].append(train_metrics["precision"])
            self.history["train_accuracy"].append(train_metrics["accuracy"])
            self.history["train_f1_score"].append(train_metrics["f1_score"])
            self.history["val_loss"].append(val_loss)
            self.history["val_dice"].append(val_metrics["dice"])
            self.history["val_iou"].append(val_metrics["iou"])
            self.history["val_sensitivity"].append(val_metrics["sensitivity"])
            self.history["val_specificity"].append(val_metrics["specificity"])
            self.history["val_precision"].append(val_metrics["precision"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["val_f1_score"].append(val_metrics["f1_score"])
            self.history["epoch_time_seconds"].append(train_time + val_time)
            self.history["train_time_seconds"].append(train_time)
            self.history["val_time_seconds"].append(val_time)
            self.history["gpu_memory_mb"].append(gpu_memory)

            print(f"  Train - Loss: {train_loss:.6f}, Dice: {train_metrics['dice']:.6f}, IoU: {train_metrics['iou']:.6f}")
            print(f"  Val   - Loss: {val_loss:.6f}, Dice: {val_metrics['dice']:.6f}, IoU: {val_metrics['iou']:.6f}")
            print(f"          Sens: {val_metrics['sensitivity']:.6f}, Prec: {val_metrics['precision']:.6f}")
            print(f"          Acc: {val_metrics['accuracy']:.6f}, F1: {val_metrics['f1_score']:.6f}")
            print(f"  ⏱️  Train: {train_time:.2f}s | Val: {val_time:.2f}s | Total: {train_time+val_time:.2f}s")
            print(f"  💾 GPU Memory: {gpu_memory:.2f} MB")

            if "csv_logger" in self.callbacks:
                row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_dice": train_metrics["dice"],
                    "train_iou": train_metrics["iou"],
                    "train_sensitivity": train_metrics["sensitivity"],
                    "train_specificity": train_metrics["specificity"],
                    "train_precision": train_metrics["precision"],
                    "train_accuracy": train_metrics["accuracy"],
                    "train_f1_score": train_metrics["f1_score"],
                    "val_loss": val_loss,
                    "val_dice": val_metrics["dice"],
                    "val_iou": val_metrics["iou"],
                    "val_sensitivity": val_metrics["sensitivity"],
                    "val_specificity": val_metrics["specificity"],
                    "val_precision": val_metrics["precision"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_f1_score": val_metrics["f1_score"],
                    "epoch_time_seconds": train_time + val_time,
                    "train_time_seconds": train_time,
                    "val_time_seconds": val_time,
                    "gpu_memory_mb": gpu_memory
                }
                self.callbacks["csv_logger"].add_row(row)

            if "model_checkpoint" in self.callbacks:
                self.callbacks["model_checkpoint"](epoch, self.model, self.optimizer, val_metrics["dice"])

            if "model_checkpoint_last" in self.callbacks:
                self.callbacks["model_checkpoint_last"](epoch, self.model, self.optimizer, epoch)

            if "early_stopping" in self.callbacks:
                if self.callbacks["early_stopping"](val_metrics["dice"]):
                    print(f"\n🛑 Early stopping activé après {epoch} époques")
                    break

            if epoch % 5 == 0 and "model_checkpoint" in self.callbacks:
                resume_path = Path(self.callbacks["model_checkpoint"].filepath).parent / "resume_checkpoint.pth"
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_score": val_metrics["dice"],
                    "history": self.history
                }
                torch.save(checkpoint, resume_path)
                print(f"  💾 Checkpoint de reprise sauvegardé")

        total_time = time.time() - total_start_time
        print("\n" + "=" * 60)
        print("ENTRAÎNEMENT SWIN UNETR TERMINÉ")
        print("=" * 60)
        print(f"Temps total: {total_time/60:.2f} minutes")

        return self.history
