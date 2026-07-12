"""
Sensor Patch Embedding for CSFAT
==================================
Embeds each sensor's time window into patch tokens.

Per-sensor 1D Conv captures local temporal dynamics,
then linear projection maps each patch to d_model dimensions.
Inspired by PatchTST (ICLR 2023, arXiv:2211.14730).

For window_size=128, patch_size=16: 8 patches per sensor.
Output shape: (B, N_sensors, N_patches, d_model)
"""

import torch
import torch.nn as nn


class SensorPatchEmbedding(nn.Module):
    """
    Per-sensor 1D convolutional patch embedding.

    Each sensor's time window is split into non-overlapping patches.
    A depthwise 1D Conv captures local temporal dynamics within patches.
    A pointwise Conv projects to d_model dimensions.

    Args:
        window_size: Input window length (must be divisible by patch_size).
        patch_size: Time steps per patch.
        d_model: Output embedding dimension.
        n_sensors: Number of sensor channels (V, I, T = 3).
        dropout: Embedding dropout rate.
    """

    def __init__(self, window_size: int = 128, patch_size: int = 16, d_model: int = 64, n_sensors: int = 3, dropout: float = 0.1):
        super().__init__()
        assert window_size % patch_size == 0, f"window_size ({window_size}) must be divisible by patch_size ({patch_size})"
        self.window_size = window_size
        self.patch_size = patch_size
        self.n_patches = window_size // patch_size
        self.d_model = d_model
        self.n_sensors = n_sensors

        # Depthwise conv: captures within-patch temporal patterns, independent per sensor
        self.depthwise_conv = nn.Conv1d(
            in_channels=n_sensors,
            out_channels=n_sensors * d_model,
            kernel_size=patch_size,
            stride=patch_size,
            groups=n_sensors,  # Depthwise: each sensor processed independently
            padding=0,
            bias=True,
        )
        self.act = nn.GELU()
        # Pointwise projection to d_model
        self.pointwise = nn.Conv1d(
            in_channels=n_sensors * d_model,
            out_channels=n_sensors * d_model,
            kernel_size=1,
            groups=n_sensors,
            bias=True,
        )
        # Learnable positional encoding: (1, N_sensors, N_patches, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_sensors, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N_sensors) raw sensor input.
        Returns:
            Patch embeddings (B, N_sensors, N_patches, d_model).
        """
        B, T, N = x.shape
        assert T == self.window_size
        # (B, T, N) -> (B, N, T) for Conv1d
        xc = x.permute(0, 2, 1)  # (B, N, T)
        out = self.act(self.depthwise_conv(xc))  # (B, N*d_model, n_patches)
        out = self.pointwise(out)                 # (B, N*d_model, n_patches)
        # Reshape to (B, N, n_patches, d_model)
        out = out.view(B, N, self.d_model, self.n_patches)
        out = out.permute(0, 1, 3, 2)            # (B, N, n_patches, d_model)
        out = out + self.pos_embed
        out = self.norm(out)
        return self.dropout(out)


class SensorTokenizer(nn.Module):
    """
    Reduces per-sensor patch embeddings to one token per sensor.
    Output: (B, N_sensors, d_model) for cross-sensor attention.

    Pool mode 'mean': average over patches (fast, effective).
    """

    def __init__(self, n_patches: int, d_model: int, n_sensors: int = 3, pool: str = "mean"):
        super().__init__()
        self.pool = pool
        if pool not in ("mean",):
            raise ValueError(f"Unsupported pool mode: {pool}. Use 'mean'.")

    def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_embeddings: (B, N_sensors, N_patches, d_model)
        Returns:
            Sensor tokens (B, N_sensors, d_model)
        """
        return patch_embeddings.mean(dim=2)  # Average over patches
