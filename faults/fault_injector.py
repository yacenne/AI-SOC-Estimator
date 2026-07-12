"""
Fault Injector
================
Applies synthetic sensor faults to multi-sensor battery windows.
Supports single-sensor and dual-sensor fault injection.
Designed for curriculum fault scheduling during CSFAT training.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .fault_types import FaultType, FaultParams, FAULT_NAMES, apply_fault, N_FAULT_CLASSES

SENSOR_V = 0
SENSOR_I = 1
SENSOR_T = 2
SENSOR_NAMES = ["Voltage", "Current", "Temperature"]
N_SENSORS = 3


@dataclass
class FaultEvent:
    """Records a single fault injection event."""
    sensor_idx: int
    fault_type: FaultType
    severity: str
    fault_start: int
    fault_end: int


@dataclass
class InjectionResult:
    """
    Result of fault injection on a multi-sensor window.
    
    Attributes:
        sensors: Corrupted sensor array (T, N_SENSORS).
        fault_labels: Per-sensor fault class labels (N_SENSORS,). 0=no fault.
        fault_mask: Boolean mask, True where fault injected (N_SENSORS,).
        events: List of FaultEvent objects.
    """
    sensors: np.ndarray
    fault_labels: np.ndarray
    fault_mask: np.ndarray
    events: List[FaultEvent] = field(default_factory=list)


class FaultInjector:
    """
    Injects synthetic sensor faults into multi-sensor battery windows.

    Args:
        fault_rate: Probability of injecting a fault into any given window.
        dual_sensor_rate: Of faulted windows, fraction with 2 sensor faults.
        severities: Severity levels to sample from.
        fault_types: FaultType values to sample from (None = all non-NONE types).
        seed: Random seed.
    """

    def __init__(
        self,
        fault_rate: float = 0.5,
        dual_sensor_rate: float = 0.15,
        severities: Optional[List[str]] = None,
        fault_types: Optional[List[FaultType]] = None,
        seed: Optional[int] = None,
    ):
        self.fault_rate = fault_rate
        self.dual_sensor_rate = dual_sensor_rate
        self.severities = severities or ["low", "medium", "high"]
        self.fault_types = fault_types or [
            FaultType.DROPOUT, FaultType.STUCK, FaultType.DRIFT,
            FaultType.SPIKE, FaultType.BIAS, FaultType.GAIN,
        ]
        self.rng = np.random.default_rng(seed)

    def set_fault_rate(self, rate: float) -> None:
        """Update fault injection probability."""
        self.fault_rate = float(np.clip(rate, 0.0, 1.0))

    def inject(
        self,
        window: np.ndarray,
        force_fault: bool = False,
        force_sensors: Optional[List[int]] = None,
        force_fault_type: Optional[FaultType] = None,
        force_severity: Optional[str] = None,
    ) -> InjectionResult:
        """
        Inject faults into a multi-sensor window.

        Args:
            window: (T, N_SENSORS) array.
            force_fault: Always inject fault if True.
            force_sensors: Inject fault on these specific sensor indices.
            force_fault_type: Use this specific fault type.
            force_severity: Use this specific severity.

        Returns:
            InjectionResult.
        """
        assert window.ndim == 2 and window.shape[1] == N_SENSORS
        corrupted = window.copy()
        fault_labels = np.zeros(N_SENSORS, dtype=np.int64)
        fault_mask = np.zeros(N_SENSORS, dtype=bool)
        events = []

        # Skip if below fault rate
        if not force_fault and float(self.rng.random()) > self.fault_rate:
            return InjectionResult(corrupted, fault_labels, fault_mask, events)

        # Choose sensors to fault
        if force_sensors is not None:
            sensors_to_fault = list(force_sensors)
        else:
            n_fault = 2 if float(self.rng.random()) < self.dual_sensor_rate else 1
            sensors_to_fault = self.rng.choice(N_SENSORS, size=n_fault, replace=False).tolist()

        for sensor_idx in sensors_to_fault:
            ft = force_fault_type if force_fault_type else FaultType(int(self.rng.choice([f.value for f in self.fault_types])))
            sev = force_severity if force_severity else str(self.rng.choice(self.severities))
            corrupted_signal, t_s, t_e = apply_fault(
                signal=corrupted[:, sensor_idx],
                fault_type=ft,
                severity=sev,
                rng=self.rng,
            )
            corrupted[:, sensor_idx] = corrupted_signal
            fault_labels[sensor_idx] = int(ft)
            fault_mask[sensor_idx] = True
            events.append(FaultEvent(sensor_idx=sensor_idx, fault_type=ft, severity=sev, fault_start=t_s, fault_end=t_e))

        return InjectionResult(corrupted, fault_labels, fault_mask, events)

    def inject_batch(self, windows: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Inject faults into a batch of windows.

        Args:
            windows: (B, T, N_SENSORS)
        Returns:
            (corrupted (B,T,N), fault_labels (B,N), fault_masks (B,N))
        """
        B = windows.shape[0]
        corrupted_all = np.zeros_like(windows)
        labels_all = np.zeros((B, N_SENSORS), dtype=np.int64)
        masks_all = np.zeros((B, N_SENSORS), dtype=bool)
        for i in range(B):
            result = self.inject(windows[i], **kwargs)
            corrupted_all[i] = result.sensors
            labels_all[i] = result.fault_labels
            masks_all[i] = result.fault_mask
        return corrupted_all, labels_all, masks_all
