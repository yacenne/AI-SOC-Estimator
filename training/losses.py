"""
Training Losses for CSFAT
============================
Implements the combined multi-task loss:

  L = L_SOC + lambda_fault * L_fault + lambda_attn * L_attn_entropy

Where:
  L_SOC:          Huber loss for SOC regression (robust to fault-induced outliers)
  L_fault:        Cross-entropy for per-sensor fault classification
  L_attn_entropy: Attention entropy regularization (encourages distributed attention)
                  Maximizes entropy -> model spreads attention across healthy sensors
                  rather than over-relying on a single dominant sensor.

Reference for attention entropy regularization:
  Encouraging diverse attention has been shown to improve robustness in
  multi-head attention (Michel et al., 2019; Correia et al., 2019).
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CSFATLoss(nn.Module):
    """
    Combined multi-task loss for CSFAT.

    Args:
        lambda_fault: Weight for fault detection cross-entropy loss.
        lambda_attn: Weight for attention entropy regularization.
        huber_delta: Delta parameter for Huber loss.
        ignore_fault_class: If True, only compute fault loss where sensor is actually faulted.
    """

    def __init__(
        self,
        lambda_fault: float = 0.3,
        lambda_attn: float = 0.01,
        huber_delta: float = 0.1,
    ):
        super().__init__()
        self.lambda_fault = lambda_fault
        self.lambda_attn = lambda_attn
        self.huber_delta = huber_delta
        self.huber = nn.HuberLoss(delta=huber_delta, reduction="mean")
        self.ce = nn.CrossEntropyLoss(reduction="mean")

    def soc_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Huber loss for SOC regression.
        Huber is more robust than MSE to fault-induced outlier SOC values.

        Args:
            pred: Predicted SOC (B, 1) or (B,).
            target: True SOC (B,).
        Returns:
            Scalar loss.
        """
        pred = pred.squeeze(-1)  # (B,)
        return self.huber(pred, target)

    def fault_loss(
        self,
        fault_logits: torch.Tensor,
        fault_labels: torch.Tensor,
        fault_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-entropy loss for per-sensor fault classification.

        Args:
            fault_logits: (B, N_sensors, n_fault_classes)
            fault_labels: (B, N_sensors) long int -- true fault classes
            fault_mask: (B, N_sensors) bool -- if provided, compute loss only on faulted sensors
                        (prevents loss from always being dominated by NONE class)
        Returns:
            Scalar loss.
        """
        B, N, C = fault_logits.shape
        # Flatten for cross-entropy: (B*N, C) and (B*N,)
        logits_flat = fault_logits.view(B * N, C)
        labels_flat = fault_labels.view(B * N)

        if fault_mask is not None:
            # Also compute loss on all positions (weighted)
            # mask_flat: True where faulted -> weight these more
            mask_flat = fault_mask.view(B * N).float()
            # Always compute CE on all, but weight faulted 2x
            weights = 1.0 + mask_flat  # 1.0 for healthy, 2.0 for faulted
            loss = F.cross_entropy(logits_flat, labels_flat, reduction="none")
            return (loss * weights).mean()
        else:
            return self.ce(logits_flat, labels_flat)

    def attention_entropy_loss(self, attn_weights: List[torch.Tensor]) -> torch.Tensor:
        """
        Attention entropy regularization: penalise LOW entropy over cross-sensor attention.

        High entropy = attention spread across all sensors (good for fault robustness).
        Low entropy  = attention collapses onto one sensor (fragile).

        We MINIMISE negative entropy, i.e. add a POSITIVE penalty when entropy is low.
        L_attn = -mean(entropy)  → minimising this maximises entropy.
        The sign is kept positive in total loss so it doesn't create negative gradients.

        Args:
            attn_weights: List of attention tensors (B, H, S, S) per layer.
        Returns:
            Scalar regularization loss >= 0 (to be minimised).
        """
        if not attn_weights:
            return torch.tensor(0.0)

        total_neg_entropy = 0.0
        count = 0
        for attn in attn_weights:
            # attn: (B, H, S, S) -- softmax weights summing to 1 along last dim
            eps = 1e-8
            # Shannon entropy per (batch, head, query token)
            entropy = -(attn * (attn + eps).log()).sum(dim=-1)  # (B, H, S)
            # We want to maximise entropy → minimise negative entropy
            total_neg_entropy = total_neg_entropy + (-entropy.mean())
            count += 1

        # Return positive loss: high value when entropy is low
        return total_neg_entropy / max(count, 1)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        soc_targets: torch.Tensor,
        fault_labels: Optional[torch.Tensor] = None,
        fault_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            outputs: Model output dict ('soc', 'fault_logits', 'attn_weights').
            soc_targets: True SOC values (B,).
            fault_labels: True fault class per sensor (B, N_sensors) long.
            fault_mask: Boolean mask (B, N_sensors) True = faulted sensor.

        Returns:
            Dict with 'total', 'soc', 'fault', 'attn' loss scalars.
        """
        # SOC loss (always computed)
        l_soc = self.soc_loss(outputs["soc"], soc_targets)

        # Fault detection loss (only when labels provided)
        l_fault = torch.tensor(0.0, device=l_soc.device)
        if fault_labels is not None and "fault_logits" in outputs and self.lambda_fault > 0:
            l_fault = self.fault_loss(outputs["fault_logits"], fault_labels, fault_mask)

        # Attention entropy regularization — disabled for now.
        # The sign of this term caused total loss to go negative in earlier runs,
        # preventing the model from learning. Re-enable once SOC training is stable.
        l_attn = torch.tensor(0.0, device=l_soc.device)

        total = l_soc + self.lambda_fault * l_fault

        return {
            "total": total,
            "soc": l_soc,
            "fault": l_fault,
            "attn": l_attn,
        }


class PretrainLoss(nn.Module):
    """
    Masked autoencoder reconstruction loss for Stage 1 pretraining.
    Computes MSE only on masked patches/tokens.

    Args:
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Masked reconstruction loss.

        Args:
            reconstructed: Model reconstruction (B, T, N) or (B, N, d).
            original: Original input (same shape).
            mask: Boolean mask (same shape), True = masked position (compute loss here).

        Returns:
            Scalar reconstruction loss on masked positions only.
        """
        diff = (reconstructed - original) ** 2
        if mask.any():
            loss = diff[mask].mean() if self.reduction == "mean" else diff[mask].sum()
        else:
            loss = diff.mean()
        return loss
