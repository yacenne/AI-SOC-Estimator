"""
CSFAT: Cross-Sensor Fault-Aware Transformer
============================================
Full model architecture for joint SOC estimation + sensor fault detection.

Architecture:
  1. SensorPatchEmbedding: per-sensor 1D Conv patch embedding (PatchTST-inspired)
  2. SensorTokenizer: aggregate patches to one token per sensor
  3. [CLS] token prepended for global SOC representation
  4. CrossSensorTransformerEncoder: attention across sensor tokens (iTransformer-inspired)
  5. SOC head: CLS -> MLP -> scalar in [0,1]
  6. Fault head: per-sensor token -> MLP -> fault class logits

Fault masking: faulty sensors are excluded from cross-sensor attention via
key_padding_mask, forcing the model to route through healthy sensors.

References:
  - iTransformer: Liu et al., ICLR 2024, arXiv:2310.06625
  - PatchTST: Nie et al., ICLR 2023, arXiv:2211.14730
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .sensor_embedding import SensorPatchEmbedding, SensorTokenizer
from .cross_sensor_transformer import CrossSensorTransformerEncoder


class CSFATEncoder(nn.Module):
    """Shared encoder used in both pretraining and fine-tuning."""

    def __init__(self, window_size=128, patch_size=16, d_model=64, n_heads=4, n_layers=3, d_ff=128, n_sensors=3, dropout=0.1):
        super().__init__()
        self.n_sensors = n_sensors
        self.d_model = d_model
        self.patch_embed = SensorPatchEmbedding(window_size, patch_size, d_model, n_sensors, dropout)
        self.tokenizer = SensorTokenizer(window_size // patch_size, d_model, n_sensors)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.transformer = CrossSensorTransformerEncoder(d_model, n_heads, n_layers, d_ff, dropout)

    def forward(self, x: torch.Tensor, sensor_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: (B, T, N_sensors)
            sensor_mask: (B, N_sensors) bool, True = faulty sensor to mask.
        Returns:
            cls_out (B, d_model), sensor_out (B, N_sensors, d_model), attn_weights list
        """
        B = x.shape[0]
        patch_emb = self.patch_embed(x)           # (B, N, n_patches, d_model)
        sensor_tokens = self.tokenizer(patch_emb)  # (B, N, d_model)
        cls = self.cls_token.expand(B, -1, -1)    # (B, 1, d_model)
        tokens = torch.cat([cls, sensor_tokens], dim=1)  # (B, N+1, d_model)
        if sensor_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            key_padding_mask = torch.cat([cls_mask, sensor_mask], dim=1)  # (B, N+1)
        else:
            key_padding_mask = None
        encoded, attn_weights = self.transformer(tokens, key_padding_mask)
        return encoded[:, 0, :], encoded[:, 1:, :], attn_weights


class CSFAT(nn.Module):
    """
    Cross-Sensor Fault-Aware Transformer (CSFAT).

    Performs:
      - SOC estimation (regression)
      - Per-sensor fault classification (7-class: None + 6 fault types)

    Args:
        window_size: Input time window length.
        patch_size: Patch size for sensor embedding.
        d_model: Embedding/hidden dimension.
        n_heads: Attention heads.
        n_layers: Transformer layers.
        d_ff: FFN hidden dimension.
        n_sensors: Number of BMS sensors (V, I, T = 3).
        n_fault_classes: Fault classes (7: NONE + DROPOUT/STUCK/DRIFT/SPIKE/BIAS/GAIN).
        dropout: Dropout rate.
    """

    def __init__(self, window_size=128, patch_size=16, d_model=64, n_heads=4, n_layers=3, d_ff=128, n_sensors=3, n_fault_classes=7, dropout=0.1):
        super().__init__()
        self.n_sensors = n_sensors
        self.n_fault_classes = n_fault_classes
        self.encoder = CSFATEncoder(window_size, patch_size, d_model, n_heads, n_layers, d_ff, n_sensors, dropout)
        self.soc_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1), nn.Sigmoid(),
        )
        self.fault_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_fault_classes),
        )

    def forward(self, x: torch.Tensor, sensor_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T, N_sensors)
            sensor_mask: (B, N_sensors) bool -- True = mask this sensor.
        Returns:
            Dict: 'soc' (B,1), 'fault_logits' (B,N_sensors,n_fault_classes),
                  'attn_weights' list, 'sensor_out' (B,N_sensors,d_model)
        """
        cls_out, sensor_out, attn_weights = self.encoder(x, sensor_mask)
        return {
            "soc": self.soc_head(cls_out),
            "fault_logits": self.fault_head(sensor_out),
            "attn_weights": attn_weights,
            "sensor_out": sensor_out,
        }

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_soc(self, x: torch.Tensor, detected_fault_sensors: Optional[List[int]] = None) -> torch.Tensor:
        """Convenience inference method with optional fault masking."""
        mask = None
        if detected_fault_sensors:
            mask = torch.zeros(x.shape[0], self.n_sensors, dtype=torch.bool, device=x.device)
            for idx in detected_fault_sensors:
                mask[:, idx] = True
        with torch.no_grad():
            return self.forward(x, mask)["soc"]
