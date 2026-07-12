"""
BMS Sensor Fault Type Definitions
====================================
Implements 6 canonical battery sensor fault types with configurable severity.

Fault taxonomy based on BMS sensor fault literature (IEEE, MDPI, 2022-2024).

Fault types:
  1. Dropout  - Complete signal loss for a duration
  2. Stuck    - Sensor freezes at its value at fault onset
  3. Drift    - Linearly growing additive error
  4. Spike    - Random high-amplitude impulse transients
  5. Bias     - Constant additive offset
  6. Gain     - Multiplicative scaling error
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional, Tuple

import numpy as np


class FaultType(IntEnum):
    """Fault class enumeration. 0 = healthy, 1-6 = fault classes."""
    NONE    = 0
    DROPOUT = 1
    STUCK   = 2
    DRIFT   = 3
    SPIKE   = 4
    BIAS    = 5
    GAIN    = 6


FAULT_NAMES = {f.value: f.name.lower() for f in FaultType}
FAULT_FROM_NAME = {v: k for k, v in FAULT_NAMES.items()}
N_FAULT_CLASSES = len(FaultType)  # 7


@dataclass
class FaultParams:
    """Parameters controlling fault severity."""
    duration_frac: float = 0.20     # Fraction of window length for duration-based faults
    rate_frac: float = 0.05         # Drift rate as fraction of signal std per timestep
    n_spikes: int = 2               # Number of spike events
    amplitude_sigma: float = 5.0    # Spike amplitude in signal std units
    bias_frac: float = 0.05         # Bias magnitude as fraction of signal std
    gain_delta: float = 0.10        # |g - 1| gain deviation from unity


# Default severity presets
SEVERITY_PRESETS: Dict[str, Dict[str, FaultParams]] = {
    "dropout": {
        "low":    FaultParams(duration_frac=0.05),
        "medium": FaultParams(duration_frac=0.20),
        "high":   FaultParams(duration_frac=0.50),
    },
    "stuck": {
        "low":    FaultParams(duration_frac=0.10),
        "medium": FaultParams(duration_frac=0.30),
        "high":   FaultParams(duration_frac=0.70),
    },
    "drift": {
        "low":    FaultParams(rate_frac=0.01),
        "medium": FaultParams(rate_frac=0.05),
        "high":   FaultParams(rate_frac=0.15),
    },
    "spike": {
        "low":    FaultParams(n_spikes=1, amplitude_sigma=2.0),
        "medium": FaultParams(n_spikes=3, amplitude_sigma=5.0),
        "high":   FaultParams(n_spikes=5, amplitude_sigma=10.0),
    },
    "bias": {
        "low":    FaultParams(bias_frac=0.02),
        "medium": FaultParams(bias_frac=0.05),
        "high":   FaultParams(bias_frac=0.15),
    },
    "gain": {
        "low":    FaultParams(gain_delta=0.02),
        "medium": FaultParams(gain_delta=0.10),
        "high":   FaultParams(gain_delta=0.25),
    },
}


def inject_dropout(signal: np.ndarray, params: FaultParams, rng: np.random.Generator) -> Tuple[np.ndarray, int, int]:
    """Dropout: zero the signal for a random contiguous duration. Model: x'(t)=0 for t in [t_s, t_e]."""
    n = len(signal)
    duration = max(1, int(n * params.duration_frac))
    t_s = int(rng.integers(0, max(1, n - duration)))
    t_e = min(n, t_s + duration)
    corrupted = signal.copy()
    corrupted[t_s:t_e] = 0.0
    return corrupted, t_s, t_e


def inject_stuck(signal: np.ndarray, params: FaultParams, rng: np.random.Generator) -> Tuple[np.ndarray, int, int]:
    """Stuck-at: sensor freezes at the value at fault onset. Model: x'(t)=x(t_s) for t in [t_s, t_e]."""
    n = len(signal)
    duration = max(1, int(n * params.duration_frac))
    t_s = int(rng.integers(0, max(1, n - duration)))
    t_e = min(n, t_s + duration)
    corrupted = signal.copy()
    corrupted[t_s:t_e] = signal[t_s]
    return corrupted, t_s, t_e


def inject_drift(signal: np.ndarray, params: FaultParams, rng: np.random.Generator) -> Tuple[np.ndarray, int, int]:
    """Linear drift: growing additive error from fault onset. Model: x'(t) = x(t) + alpha*(t-t_s)."""
    n = len(signal)
    sig_std = float(np.std(signal)) + 1e-8
    alpha = params.rate_frac * sig_std
    t_s = int(rng.integers(0, max(1, n // 2)))
    corrupted = signal.copy().astype(float)
    drift = alpha * np.arange(0, n - t_s)
    if rng.random() < 0.5:  # Random drift direction
        drift = -drift
    corrupted[t_s:] += drift
    return corrupted, t_s, n


def inject_spike(signal: np.ndarray, params: FaultParams, rng: np.random.Generator) -> Tuple[np.ndarray, int, int]:
    """Spike: random high-amplitude impulses. Model: x'(t_k) = x(t_k) + A_k."""
    n = len(signal)
    sig_std = float(np.std(signal)) + 1e-8
    corrupted = signal.copy().astype(float)
    n_spikes = min(params.n_spikes, n)
    spike_times = rng.choice(n, size=n_spikes, replace=False)
    amplitudes = rng.normal(0, params.amplitude_sigma * sig_std, size=n_spikes)
    corrupted[spike_times] += amplitudes
    t_s = int(spike_times.min())
    t_e = int(spike_times.max()) + 1
    return corrupted, t_s, t_e


def inject_bias(signal: np.ndarray, params: FaultParams, rng: np.random.Generator) -> Tuple[np.ndarray, int, int]:
    """Bias: constant additive offset. Model: x'(t) = x(t) + b."""
    sig_std = float(np.std(signal)) + 1e-8
    b_max = params.bias_frac * sig_std
    b = float(rng.uniform(-b_max, b_max))
    return signal.copy() + b, 0, len(signal)


def inject_gain(signal: np.ndarray, params: FaultParams, rng: np.random.Generator) -> Tuple[np.ndarray, int, int]:
    """Gain: multiplicative scaling error. Model: x'(t) = g * x(t)."""
    direction = rng.choice([-1, 1])
    g = 1.0 + float(direction) * params.gain_delta
    return signal.copy() * g, 0, len(signal)


FAULT_INJECT_FNS = {
    FaultType.DROPOUT: inject_dropout,
    FaultType.STUCK:   inject_stuck,
    FaultType.DRIFT:   inject_drift,
    FaultType.SPIKE:   inject_spike,
    FaultType.BIAS:    inject_bias,
    FaultType.GAIN:    inject_gain,
}


def apply_fault(
    signal: np.ndarray,
    fault_type: FaultType,
    severity: str = "medium",
    params: Optional[FaultParams] = None,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, int, int]:
    """
    Apply a fault to a 1D sensor signal.

    Args:
        signal: 1D numpy array (raw sensor values).
        fault_type: FaultType enum value.
        severity: 'low', 'medium', or 'high'.
        params: Optional custom FaultParams (overrides severity preset).
        rng: Random number generator for reproducibility.
        seed: Optional seed if rng is None.

    Returns:
        Tuple of (corrupted_signal, fault_start_idx, fault_end_idx).
    """
    if fault_type == FaultType.NONE:
        return signal.copy(), 0, 0
    if rng is None:
        rng = np.random.default_rng(seed)
    if params is None:
        fault_name = FAULT_NAMES[int(fault_type)]
        params = SEVERITY_PRESETS[fault_name][severity]
    return FAULT_INJECT_FNS[fault_type](signal, params, rng)
