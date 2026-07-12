"""
Baseline Models for CSFAT Comparison
=======================================
Standard baselines for ablation and benchmarking:
  1. VanillaTransformer - Standard temporal Transformer, flat input
  2. LSTMBaseline - LSTM or BiLSTM
  3. build_model - Factory function for all models
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class VanillaTransformer(nn.Module):
    """
    Standard temporal Transformer. Concatenates all sensors as features per timestep.
    No fault-awareness; serves as ablation baseline for the cross-sensor design.
    """

    def __init__(self, window_size=128, n_sensors=3, d_model=64, n_heads=4, n_layers=3, d_ff=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_sensors, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, window_size, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.soc_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        h = self.input_proj(x) + self.pos_embed
        h = self.norm(self.transformer(h))
        return {"soc": self.soc_head(h.mean(dim=1)), "attn_weights": []}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LSTMBaseline(nn.Module):
    """Bidirectional LSTM baseline for SOC estimation."""

    def __init__(self, n_sensors=3, hidden_size=64, n_layers=2, dropout=0.1, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(n_sensors, hidden_size, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0, bidirectional=bidirectional)
        out_size = hidden_size * (2 if bidirectional else 1)
        self.soc_head = nn.Sequential(
            nn.Linear(out_size, out_size // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(out_size // 2, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        out, _ = self.lstm(x)
        return {"soc": self.soc_head(out[:, -1, :]), "attn_weights": []}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(model_name: str, config: dict) -> nn.Module:
    """
    Build a model by name from config.

    Args:
        model_name: 'csfat', 'vanilla_transformer', 'lstm', or 'bilstm'.
        config: Loaded config.yaml dict.
    Returns:
        Instantiated model.
    """
    from .csfat import CSFAT
    mc, wc = config["model"], config["window"]
    if model_name == "csfat":
        return CSFAT(wc["size"], wc["patch_size"], mc["d_model"], mc["n_heads"], mc["n_layers"],
                     mc["d_ff"], mc["n_sensors"], mc["n_fault_classes"], mc["dropout"])
    elif model_name == "vanilla_transformer":
        return VanillaTransformer(wc["size"], mc["n_sensors"], mc["d_model"], mc["n_heads"],
                                  mc["n_layers"], mc["d_ff"], mc["dropout"])
    elif model_name in ("lstm", "bilstm"):
        return LSTMBaseline(mc["n_sensors"], mc["d_model"], mc["n_layers"],
                            mc["dropout"], bidirectional=(model_name == "bilstm"))
    else:
        raise ValueError(f"Unknown model: {model_name}")
