"""
CSFAT Fine-tuning Trainer
===========================
Stage 2: Supervised fine-tuning for joint SOC estimation + fault detection.

Uses curriculum fault injection scheduling:
  Phase 1 (warmup):  Train on clean data only (stable SOC baseline)
  Phase 2 (ramp):    Gradually increase fault injection rate
  Phase 3 (steady):  Train at full fault rate with dual-sensor faults

Supports loading a Stage 1 pretrained encoder checkpoint.
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import CSFATLoss


class Trainer:
    """
    Fine-tuning trainer for CSFAT.

    Args:
        model: CSFAT model (or baseline).
        config: Loaded config.yaml dict.
        device: Torch device (auto-detect if None).
        checkpoint_dir: Where to save model checkpoints.
        log_dir: Where to save training logs.
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "./checkpoints",
        log_dir: str = "./logs",
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        ft_cfg = config["finetune"]
        self.optimizer = AdamW(
            model.parameters(),
            lr=ft_cfg["lr"],
            weight_decay=ft_cfg["weight_decay"],
        )
        self.criterion = CSFATLoss(
            lambda_fault=ft_cfg["lambda_fault"],
            lambda_attn=ft_cfg["lambda_attn"],
        )
        self.n_epochs = ft_cfg["epochs"]

        print(f"Trainer initialized | Device: {self.device}")
        if hasattr(model, "count_parameters"):
            print(f"Model parameters: {model.count_parameters():,}")

    def _train_epoch(
        self,
        loader: DataLoader,
        fault_injector,
        epoch: int,
    ) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = total_soc = total_fault = 0.0
        n_batches = 0

        for batch in loader:
            sensors = batch["sensors"].to(self.device)       # (B, T, N)
            soc_target = batch["soc"].to(self.device)        # (B,)
            fault_labels = batch.get("fault_labels")
            fault_mask = batch.get("fault_mask")
            if fault_labels is not None:
                fault_labels = fault_labels.to(self.device)  # (B, N)
            if fault_mask is not None:
                fault_mask = fault_mask.to(self.device)      # (B, N) bool

            # Build sensor_mask for model (mask faulted sensors from attention)
            model_sensor_mask = fault_mask if fault_mask is not None else None

            self.optimizer.zero_grad()
            outputs = self.model(sensors, model_sensor_mask)
            losses = self.criterion(outputs, soc_target, fault_labels, fault_mask)

            losses["total"].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += losses["total"].item()
            total_soc += losses["soc"].item()
            total_fault += losses["fault"].item()
            n_batches += 1

        return {
            "loss": total_loss / max(n_batches, 1),
            "soc_loss": total_soc / max(n_batches, 1),
            "fault_loss": total_fault / max(n_batches, 1),
        }

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Run one validation epoch (clean data, no fault injection)."""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []

        for batch in loader:
            sensors = batch["sensors"].to(self.device)
            soc_target = batch["soc"].to(self.device)
            outputs = self.model(sensors, sensor_mask=None)
            losses = self.criterion(outputs, soc_target)
            total_loss += losses["total"].item()
            all_preds.append(outputs["soc"].squeeze(-1).cpu())
            all_targets.append(soc_target.cpu())

        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        rmse = torch.sqrt(((preds - targets) ** 2).mean()).item()
        mae = (preds - targets).abs().mean().item()

        return {
            "val_loss": total_loss / max(len(loader), 1),
            "val_rmse": rmse,
            "val_mae": mae,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        fault_injector=None,
        curriculum_scheduler=None,
        run_name: str = "csfat",
    ) -> Dict[str, list]:
        """
        Run full fine-tuning training loop.

        Args:
            train_loader: Training DataLoader (with fault injection via Dataset).
            val_loader: Validation DataLoader (clean data).
            fault_injector: FaultInjector to update rate each epoch (optional).
            curriculum_scheduler: CurriculumScheduler for fault rate scheduling.
            run_name: Name prefix for saved checkpoints.

        Returns:
            Training history dict with per-epoch metrics.
        """
        ft_cfg = self.config["finetune"]
        scheduler = CosineAnnealingLR(self.optimizer, T_max=self.n_epochs, eta_min=1e-6)

        history = {
            "train_loss": [], "val_rmse": [], "val_mae": [],
            "soc_loss": [], "fault_loss": [], "fault_rate": [],
        }
        best_val_rmse = float("inf")

        print(f"\n{'='*60}")
        print(f"Starting Fine-tuning: {self.n_epochs} epochs")
        print(f"{'='*60}")

        for epoch in range(self.n_epochs):
            # Update fault rate via curriculum
            if curriculum_scheduler is not None and fault_injector is not None:
                rate = curriculum_scheduler.get_fault_rate(epoch)
                fault_injector.set_fault_rate(rate)
                # Also update the dataset's fault injector rate
                if hasattr(train_loader.dataset, "fault_injector") and train_loader.dataset.fault_injector is not None:
                    train_loader.dataset.fault_injector.set_fault_rate(rate)
            else:
                rate = 0.0

            t0 = time.time()
            train_metrics = self._train_epoch(train_loader, fault_injector, epoch)
            val_metrics = self._val_epoch(val_loader)
            scheduler.step()
            elapsed = time.time() - t0

            # Record history
            history["train_loss"].append(train_metrics["loss"])
            history["soc_loss"].append(train_metrics["soc_loss"])
            history["fault_loss"].append(train_metrics["fault_loss"])
            history["val_rmse"].append(val_metrics["val_rmse"])
            history["val_mae"].append(val_metrics["val_mae"])
            history["fault_rate"].append(rate)

            # Save best model
            if val_metrics["val_rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["val_rmse"]
                ckpt_path = self.checkpoint_dir / f"{run_name}_best.pt"
                torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "val_rmse": best_val_rmse}, ckpt_path)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"Epoch [{epoch+1:3d}/{self.n_epochs}] "
                    f"Loss: {train_metrics['loss']:.4f} | "
                    f"SOC: {train_metrics['soc_loss']:.4f} | "
                    f"Val RMSE: {val_metrics['val_rmse']*100:.3f}% | "
                    f"Fault Rate: {rate:.2f} | "
                    f"Time: {elapsed:.1f}s"
                )

        print(f"\nBest Val RMSE: {best_val_rmse*100:.3f}%")
        print(f"Checkpoint saved to: {self.checkpoint_dir / f'{run_name}_best.pt'}")
        return history

    def load_pretrained_encoder(self, checkpoint_path: str) -> None:
        """
        Load encoder weights from a Stage 1 MAE pretraining checkpoint.
        Only loads the encoder sub-module, leaving heads randomly initialized.

        Args:
            checkpoint_path: Path to pretrain checkpoint .pt file.
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt)
        # Filter to encoder-only weights
        encoder_state = {k.replace("encoder.", ""): v for k, v in state.items() if k.startswith("encoder.")}
        if hasattr(self.model, "encoder"):
            missing, unexpected = self.model.encoder.load_state_dict(encoder_state, strict=False)
            print(f"Loaded pretrained encoder. Missing: {missing}, Unexpected: {unexpected}")
        else:
            print("[WARN] Model has no 'encoder' attribute -- loading full checkpoint")
            self.model.load_state_dict(state, strict=False)
