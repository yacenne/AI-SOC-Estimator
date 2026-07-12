"""
Cross-Sensor Transformer Encoder for CSFAT
============================================
Applies self-attention ACROSS sensor tokens (not across time steps).

Inspired by iTransformer (ICLR 2024, arXiv:2310.06625) which applies
attention across variates rather than timesteps.

Key insight: By treating each sensor as a token, attention can learn
  which sensors compensate for others when a sensor is faulty/masked.

Input:  (B, N_sensors + 1, d_model) -- N sensor tokens + 1 [CLS] token
Output: (B, N_sensors + 1, d_model)
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossSensorAttention(nn.Module):
    """
    Multi-head self-attention across sensor tokens.
    
    The attention matrix is (N_sensors+1) x (N_sensors+1) -- tiny and fast.
    key_padding_mask=True for faulty sensors excludes them from key/value.
    
    Args:
        d_model: Embedding dimension.
        n_heads: Attention heads.
        dropout: Attention weight dropout.
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, S, d_model) where S = N_sensors + 1.
            key_padding_mask: (B, S) bool, True = masked sensor.
        Returns:
            (output (B,S,d_model), attn_weights (B,H,S,S))
        """
        B, S, D = x.shape
        H, Dh = self.n_heads, self.d_head
        Q = self.q(x).view(B, S, H, Dh).transpose(1, 2)  # (B,H,S,Dh)
        K = self.k(x).view(B, S, H, Dh).transpose(1, 2)
        V = self.v(x).view(B, S, H, Dh).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B,H,S,S)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = self.drop(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, S, D)
        return self.out(out), attn


class CrossSensorTransformerLayer(nn.Module):
    """Single cross-sensor Transformer layer (pre-norm)."""

    def __init__(self, d_model: int = 64, n_heads: int = 4, d_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = CrossSensorAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        normed = self.norm1(x)
        attn_out, attn_w = self.attn(normed, key_padding_mask)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, attn_w


class CrossSensorTransformerEncoder(nn.Module):
    """Stack of cross-sensor Transformer layers."""

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 3, d_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([CrossSensorTransformerLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, S, d_model)
            key_padding_mask: (B, S) bool
        Returns:
            (encoded (B,S,d_model), list of attn_weights per layer)
        """
        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x, key_padding_mask)
            attn_maps.append(attn)
        return self.norm(x), attn_maps
