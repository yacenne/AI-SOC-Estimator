"""
CSFAT Dataset and DataLoader
==============================
Sliding window PyTorch Dataset for multi-sensor battery SOC data.
Supports:
  - Sliding window extraction from time-series DataFrames
  - Per-sensor normalization (z-score, fit on training set)
  - On-the-fly fault injection during training
  - Train/val/test split utilities

Expected DataFrame columns: Time, Voltage, Current, Temperature, SOC
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


SENSOR_COLS = ["Voltage", "Current", "Temperature"]
LABEL_COL = "SOC"


class SensorNormalizer:
    """
    Per-sensor z-score normalizer. Fit on training data, applied to all splits.

    Args:
        eps: Small constant to avoid division by zero.
    """

    def __init__(self, eps: float = 1e-8):
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None
        self.eps = eps

    def fit(self, windows: np.ndarray) -> "SensorNormalizer":
        """
        Fit normalizer on a collection of windows.

        Args:
            windows: Array of shape (N, T, n_sensors) or (N*T, n_sensors).
        Returns:
            self (for chaining).
        """
        if windows.ndim == 3:
            flat = windows.reshape(-1, windows.shape[-1])
        else:
            flat = windows
        self.means = flat.mean(axis=0)
        self.stds = flat.std(axis=0) + self.eps
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        """Normalize windows using fitted mean/std."""
        assert self.means is not None, "Call fit() before transform()"
        return (windows - self.means) / self.stds

    def inverse_transform(self, windows: np.ndarray) -> np.ndarray:
        """Inverse normalize."""
        return windows * self.stds + self.means

    def fit_transform(self, windows: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(windows).transform(windows)


def extract_windows(
    df: pd.DataFrame,
    window_size: int = 128,
    stride: int = 32,
    sensor_cols: List[str] = SENSOR_COLS,
    label_col: str = LABEL_COL,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract sliding windows from a time-series DataFrame.

    Args:
        df: DataFrame with sensor columns and SOC label.
        window_size: Number of time steps per window.
        stride: Sliding step between windows.
        sensor_cols: List of sensor column names.
        label_col: SOC label column name.

    Returns:
        Tuple of:
          - windows: (N, window_size, n_sensors) float32
          - labels: (N,) float32 SOC at last timestep of each window
    """
    required = sensor_cols + [label_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {df.columns.tolist()}")

    data = df[sensor_cols].values.astype(np.float32)
    soc = df[label_col].values.astype(np.float32)
    n = len(data)

    windows, labels = [], []
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        windows.append(data[start:end])      # (window_size, n_sensors)
        labels.append(soc[end - 1])          # SOC at last timestep

    if not windows:
        return np.empty((0, window_size, len(sensor_cols)), dtype=np.float32), np.empty(0, dtype=np.float32)

    return np.stack(windows, axis=0), np.array(labels, dtype=np.float32)


def build_windows_from_dfs(
    dfs: List[pd.DataFrame],
    window_size: int = 128,
    stride: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract windows from a list of DataFrames (e.g., multiple drive cycles).

    Returns concatenated windows and labels arrays.
    """
    all_windows, all_labels = [], []
    for df in dfs:
        if df is None or len(df) < window_size:
            continue
        w, l = extract_windows(df, window_size, stride)
        if len(w) > 0:
            all_windows.append(w)
            all_labels.append(l)

    if not all_windows:
        n_sensors = len(SENSOR_COLS)
        return np.empty((0, window_size, n_sensors), dtype=np.float32), np.empty(0, dtype=np.float32)

    return np.concatenate(all_windows, axis=0), np.concatenate(all_labels, axis=0)


class BatterySOCDataset(Dataset):
    """
    PyTorch Dataset for battery SOC estimation with optional fault injection.

    Args:
        windows: Sensor windows (N, T, n_sensors).
        labels: SOC labels (N,).
        fault_injector: Optional FaultInjector for on-the-fly augmentation.
        normalizer: Optional SensorNormalizer (applied before fault injection).
        return_fault_info: If True, return fault labels and masks alongside windows.
    """

    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        fault_injector=None,
        normalizer: Optional[SensorNormalizer] = None,
        return_fault_info: bool = False,
    ):
        assert len(windows) == len(labels), "windows and labels must have same length"
        self.windows = windows.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.fault_injector = fault_injector
        self.normalizer = normalizer
        self.return_fault_info = return_fault_info

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        window = self.windows[idx].copy()  # (T, n_sensors)
        soc = self.labels[idx]

        # Normalize sensors
        if self.normalizer is not None:
            window = self.normalizer.transform(window)

        fault_labels = np.zeros(window.shape[1], dtype=np.int64)
        fault_mask = np.zeros(window.shape[1], dtype=bool)

        # Apply fault injection
        if self.fault_injector is not None:
            result = self.fault_injector.inject(window)
            window = result.sensors
            fault_labels = result.fault_labels
            fault_mask = result.fault_mask

        item = {
            "sensors": torch.tensor(window, dtype=torch.float32),
            "soc": torch.tensor(soc, dtype=torch.float32),
        }
        if self.return_fault_info:
            item["fault_labels"] = torch.tensor(fault_labels, dtype=torch.long)
            item["fault_mask"] = torch.tensor(fault_mask, dtype=torch.bool)

        return item


def build_dataloaders(
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    val_windows: np.ndarray,
    val_labels: np.ndarray,
    test_windows: np.ndarray,
    test_labels: np.ndarray,
    batch_size: int = 64,
    num_workers: int = 0,
    fault_injector=None,
    return_fault_info: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, SensorNormalizer]:
    """
    Build train/val/test DataLoaders with normalization and optional fault injection.

    Args:
        train_windows, train_labels: Training split.
        val_windows, val_labels: Validation split.
        test_windows, test_labels: Test split.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        fault_injector: FaultInjector for training augmentation (val/test are clean).
        return_fault_info: Include fault labels/masks in batches.

    Returns:
        (train_loader, val_loader, test_loader, normalizer)
    """
    # Fit normalizer on training data
    normalizer = SensorNormalizer()
    normalizer.fit(train_windows)

    train_ds = BatterySOCDataset(train_windows, train_labels, fault_injector, normalizer, return_fault_info)
    val_ds   = BatterySOCDataset(val_windows,   val_labels,   None,           normalizer, return_fault_info)
    test_ds  = BatterySOCDataset(test_windows,  test_labels,  None,           normalizer, return_fault_info)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, normalizer
