"""
CSFAT Robustness Evaluator
============================
Runs the full fault robustness evaluation protocol:
  - Tests all 6 fault types x 3 severity levels x 3 sensors
  - Computes RMSE, MAE, MaxError, Degradation Ratio per experiment
  - Optionally evaluates fault detection head (F1, precision, recall)

This is the main evaluation harness for the paper's Table 1 and Figure 2.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import SOCMetrics, FaultMetrics, compute_degradation_ratio, summarize_robustness_results
from faults.fault_types import FaultType, FAULT_NAMES
try:
    from faults.fault_types import SENSOR_NAMES
except ImportError:
    SENSOR_NAMES = ["Voltage", "Current", "Temperature"]


SENSOR_NAMES_DEFAULT = ["Voltage", "Current", "Temperature"]
SEVERITIES = ["low", "medium", "high"]
FAULT_TYPES = [FaultType.DROPOUT, FaultType.STUCK, FaultType.DRIFT, FaultType.SPIKE, FaultType.BIAS, FaultType.GAIN]


class RobustnessEvaluator:
    """
    Evaluates CSFAT robustness across all fault types, severities, and sensor combinations.

    Args:
        model: Trained CSFAT model (or baseline).
        device: Torch device.
        sensor_names: Names for each sensor channel.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        sensor_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.sensor_names = sensor_names or SENSOR_NAMES_DEFAULT

    @torch.no_grad()
    def evaluate_clean(self, loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate on clean (no-fault) test data.

        Returns:
            Dict with rmse, mae, max_error (clean baseline).
        """
        all_preds, all_targets = [], []
        for batch in loader:
            sensors = batch["sensors"].to(self.device)
            targets = batch["soc"].cpu().numpy()
            preds = self.model(sensors)["soc"].squeeze(-1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets)

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        metrics = SOCMetrics.compute_all(preds, targets)
        metrics["label"] = "clean"
        return metrics

    @torch.no_grad()
    def evaluate_faulted(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        fault_type: FaultType,
        severity: str,
        sensor_idx: int,
        normalizer=None,
        batch_size: int = 64,
        use_masking: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate model on data with a specific fault injected.

        Args:
            windows: (N, T, n_sensors) clean sensor windows.
            labels: (N,) SOC labels.
            fault_type: FaultType to inject.
            severity: 'low', 'medium', or 'high'.
            sensor_idx: Which sensor to fault (0=V, 1=I, 2=T).
            normalizer: SensorNormalizer (if used during training).
            batch_size: Inference batch size.
            use_masking: If True, tell model which sensor is faulted (oracle masking).

        Returns:
            Dict with rmse, mae, max_error, degradation_ratio metrics.
        """
        from faults.fault_types import apply_fault
        from faults.fault_injector import FaultInjector

        injector = FaultInjector(fault_rate=1.0, seed=42)
        all_preds, all_targets = [], []

        for i in range(0, len(windows), batch_size):
            batch_w = windows[i:i+batch_size].copy()
            batch_l = labels[i:i+batch_size]

            # Inject fault on the specified sensor for all windows in batch
            for j in range(len(batch_w)):
                corrupted, _, _ = apply_fault(
                    batch_w[j, :, sensor_idx], fault_type, severity, seed=42+j
                )
                batch_w[j, :, sensor_idx] = corrupted

            # Normalize
            if normalizer is not None:
                batch_w = normalizer.transform(batch_w)

            x = torch.tensor(batch_w, dtype=torch.float32).to(self.device)

            # Build sensor mask for model (oracle: we know which sensor is faulted)
            sensor_mask = None
            if use_masking:
                sensor_mask = torch.zeros(len(batch_w), len(self.sensor_names), dtype=torch.bool, device=self.device)
                sensor_mask[:, sensor_idx] = True

            preds = self.model(x, sensor_mask)["soc"].squeeze(-1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(batch_l)

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        metrics = SOCMetrics.compute_all(preds, targets)
        metrics["fault_type"] = FAULT_NAMES.get(int(fault_type), str(fault_type))
        metrics["severity"] = severity
        metrics["sensor"] = self.sensor_names[sensor_idx]
        return metrics

    def run_full_protocol(
        self,
        clean_loader: DataLoader,
        test_windows: np.ndarray,
        test_labels: np.ndarray,
        normalizer=None,
        fault_types: Optional[List[FaultType]] = None,
        severities: Optional[List[str]] = None,
        sensor_indices: Optional[List[int]] = None,
        use_masking: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run the full 6 x 3 x 3 robustness evaluation protocol.

        Args:
            clean_loader: DataLoader for clean test data (to get baseline RMSE).
            test_windows: (N, T, n_sensors) clean windows for fault injection.
            test_labels: (N,) SOC labels.
            normalizer: SensorNormalizer fitted on training data.
            fault_types: Fault types to test (default: all 6).
            severities: Severity levels (default: low/medium/high).
            sensor_indices: Sensor channels to test (default: all 3).
            use_masking: Use oracle sensor masking during evaluation.

        Returns:
            Dict: experiment_key -> metrics dict.
            Key format: "{fault_type}_{severity}_{sensor_name}"
        """
        if fault_types is None:
            fault_types = FAULT_TYPES
        if severities is None:
            severities = SEVERITIES
        if sensor_indices is None:
            sensor_indices = list(range(len(self.sensor_names)))

        results = {}

        # Baseline: clean data
        print("Evaluating on clean data...")
        clean_metrics = self.evaluate_clean(clean_loader)
        results["clean"] = clean_metrics
        baseline_rmse = clean_metrics["rmse"]
        print(f"  Clean RMSE: {baseline_rmse*100:.3f}%")

        # Fault injection experiments
        total = len(fault_types) * len(severities) * len(sensor_indices)
        print(f"\nRunning {total} fault injection experiments...")

        with tqdm(total=total, desc="Fault Evaluation") as pbar:
            for ft in fault_types:
                for sev in severities:
                    for s_idx in sensor_indices:
                        key = f"{FAULT_NAMES[int(ft)]}_{sev}_{self.sensor_names[s_idx]}"
                        try:
                            metrics = self.evaluate_faulted(
                                test_windows, test_labels, ft, sev, s_idx, normalizer, use_masking=use_masking
                            )
                            metrics["degradation_ratio"] = compute_degradation_ratio(baseline_rmse, metrics["rmse"])
                        except Exception as e:
                            print(f"\n[WARN] {key}: {e}")
                            metrics = {"rmse": float("nan"), "mae": float("nan"), "degradation_ratio": float("nan")}
                        results[key] = metrics
                        pbar.update(1)

        print(summarize_robustness_results(results))
        return results
