"""
CSFAT Evaluation Metrics
===========================
Computes SOC estimation metrics and fault robustness metrics.

SOC metrics:
  - RMSE, MAE, MaxError (standard regression metrics)
  - Degradation Ratio (DR): RMSE_faulted / RMSE_clean (lower = more robust)

Fault detection metrics:
  - Precision, Recall, F1 per fault class and per sensor
  - Detection latency (timesteps to first detection)

Usage:
    from evaluation.metrics import SOCMetrics, FaultMetrics, compute_degradation_ratio
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)


class SOCMetrics:
    """
    Container for SOC estimation evaluation metrics.
    All SOC values should be in [0, 1] (not percentage).
    """

    @staticmethod
    def rmse(pred: np.ndarray, target: np.ndarray) -> float:
        """Root Mean Square Error (percentage points if multiplied by 100)."""
        return float(np.sqrt(np.mean((pred - target) ** 2)))

    @staticmethod
    def mae(pred: np.ndarray, target: np.ndarray) -> float:
        """Mean Absolute Error."""
        return float(np.mean(np.abs(pred - target)))

    @staticmethod
    def max_error(pred: np.ndarray, target: np.ndarray) -> float:
        """Maximum absolute error."""
        return float(np.max(np.abs(pred - target)))

    @staticmethod
    def compute_all(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Compute all SOC metrics and return as dict."""
        m = SOCMetrics
        return {
            "rmse": m.rmse(pred, target),
            "rmse_pct": m.rmse(pred, target) * 100,   # As percentage
            "mae": m.mae(pred, target),
            "mae_pct": m.mae(pred, target) * 100,
            "max_error": m.max_error(pred, target),
            "max_error_pct": m.max_error(pred, target) * 100,
        }


def compute_degradation_ratio(
    rmse_clean: float,
    rmse_faulted: float,
    eps: float = 1e-8,
) -> float:
    """
    Degradation Ratio (DR) = RMSE_faulted / RMSE_clean.
    Lower is better. DR=1.0 means no degradation from fault.
    DR=2.0 means fault doubled the error.

    Args:
        rmse_clean: RMSE on clean (no-fault) test data.
        rmse_faulted: RMSE on fault-injected test data.
        eps: Small constant to avoid division by zero.
    Returns:
        Degradation ratio (float).
    """
    return rmse_faulted / (rmse_clean + eps)


class FaultMetrics:
    """
    Metrics for per-sensor fault detection evaluation.

    fault_labels and pred_labels are per-sensor class indices (0=NONE, 1-6=fault).
    """

    @staticmethod
    def per_sensor_f1(
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        n_sensors: int = 3,
        sensor_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute per-sensor macro F1 scores.

        Args:
            true_labels: (N, n_sensors) int array of true fault classes.
            pred_labels: (N, n_sensors) int array of predicted fault classes.
            n_sensors: Number of sensors.
            sensor_names: Names for each sensor.

        Returns:
            Dict: sensor_name -> macro F1 score.
        """
        names = sensor_names or [f"sensor_{i}" for i in range(n_sensors)]
        results = {}
        for i, name in enumerate(names):
            y_true = true_labels[:, i]
            y_pred = pred_labels[:, i]
            # Binary: fault detected (any fault) vs no fault
            binary_true = (y_true > 0).astype(int)
            binary_pred = (y_pred > 0).astype(int)
            results[f"{name}_binary_f1"] = float(f1_score(binary_true, binary_pred, zero_division=0))
            results[f"{name}_precision"] = float(precision_score(binary_true, binary_pred, zero_division=0))
            results[f"{name}_recall"] = float(recall_score(binary_true, binary_pred, zero_division=0))
        return results

    @staticmethod
    def multiclass_report(
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        fault_names: Optional[List[str]] = None,
    ) -> str:
        """
        Full classification report for multi-class fault type detection.
        Flattens all sensors together.

        Args:
            true_labels: (N, n_sensors) true class indices.
            pred_labels: (N, n_sensors) predicted class indices.
            fault_names: Class names (default: NONE, DROPOUT, ...).

        Returns:
            Classification report string.
        """
        names = fault_names or ["NONE", "DROPOUT", "STUCK", "DRIFT", "SPIKE", "BIAS", "GAIN"]
        flat_true = true_labels.flatten()
        flat_pred = pred_labels.flatten()
        present = np.unique(np.concatenate([flat_true, flat_pred]))
        present_names = [names[i] for i in present if i < len(names)]
        return classification_report(flat_true, flat_pred, labels=present, target_names=present_names, zero_division=0)

    @staticmethod
    def confusion_matrix(
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
    ) -> np.ndarray:
        """Compute confusion matrix (flattened across all sensors)."""
        return confusion_matrix(true_labels.flatten(), pred_labels.flatten())


def summarize_robustness_results(
    results: Dict[str, Dict[str, float]],
) -> str:
    """
    Format a robustness evaluation results dict into a readable table.

    Args:
        results: Dict of {experiment_key -> {metric -> value}}.
    Returns:
        Formatted string table.
    """
    lines = ["\n" + "=" * 70, "ROBUSTNESS EVALUATION SUMMARY", "=" * 70]
    lines.append(f"{'Experiment':<35} {'RMSE%':>8} {'MAE%':>8} {'DR':>8}")
    lines.append("-" * 70)
    for key, metrics in sorted(results.items()):
        rmse_pct = metrics.get("rmse_pct", metrics.get("rmse", 0) * 100)
        mae_pct = metrics.get("mae_pct", metrics.get("mae", 0) * 100)
        dr = metrics.get("degradation_ratio", float("nan"))
        lines.append(f"{key:<35} {rmse_pct:>8.3f} {mae_pct:>8.3f} {dr:>8.3f}")
    lines.append("=" * 70)
    return "\n".join(lines)
