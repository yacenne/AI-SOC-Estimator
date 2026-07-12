"""
Panasonic 18650PF Dataset Loader
==================================

Dataset DOI: 10.17632/wykht8y7tg.1
Download: https://data.mendeley.com/datasets/wykht8y7tg/1

Actual directory structure (as downloaded):
  data/raw/Panasonic 18650PF Data/Panasonic 18650PF Data/
    25degC/
      Drive cycles/     <- .mat files here
      1C discharge tests_end_of_tests/
      ...
    10degC/
    0degC/
   -10degC/
   -20degC/

Files are MATLAB structured arrays with field:
  meas.Voltage, meas.Current, meas.Battery_Temp_degC,
  meas.Time, meas.Ah (cumulative Ah)

SOC is computed from meas.Ah (Coulomb counting already done by test rig).
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import interpolate


SENSOR_COLS = ["Voltage", "Current", "Temperature"]

TEMP_DIR_MAP = {
    25:  "25degC",
    10:  "10degC",
    0:   "0degC",
    -10: "-10degC",
    -20: "-20degC",
}


def _extract_mat_field(meas, field: str) -> np.ndarray:
    """Safely extract a 1-D array from the meas structured array."""
    try:
        arr = meas[field][0, 0].flatten().astype(float)
        return arr
    except Exception:
        return np.array([])


def load_mat_file(filepath: str, capacity_ah: float = 2.9) -> Optional[pd.DataFrame]:
    """
    Load a single Panasonic .mat file and return a clean DataFrame
    with columns [Time, Voltage, Current, Temperature, SOC].
    """
    try:
        mat = sio.loadmat(filepath)
    except Exception as e:
        print(f"  [WARN] Could not load {os.path.basename(filepath)}: {e}")
        return None

    if "meas" not in mat:
        return None

    meas = mat["meas"]

    time = _extract_mat_field(meas, "Time")
    volt = _extract_mat_field(meas, "Voltage")
    curr = _extract_mat_field(meas, "Current")
    temp = _extract_mat_field(meas, "Battery_Temp_degC")
    ah   = _extract_mat_field(meas, "Ah")

    # Basic sanity check
    min_len = min(len(time), len(volt), len(curr), len(temp), len(ah))
    if min_len < 50:
        return None

    time = time[:min_len]
    volt = volt[:min_len]
    curr = curr[:min_len]
    temp = temp[:min_len]
    ah   = ah[:min_len]

    # Remove NaNs
    mask = np.isfinite(time) & np.isfinite(volt) & np.isfinite(curr) & np.isfinite(temp)
    if mask.sum() < 50:
        return None
    time, volt, curr, temp, ah = (
        time[mask], volt[mask], curr[mask], temp[mask], ah[mask]
    )

    # Sort by time
    idx = np.argsort(time)
    time, volt, curr, temp, ah = (
        time[idx], volt[idx], curr[idx], temp[idx], ah[idx]
    )

    # Resample to 1 Hz
    t_new = np.arange(time[0], time[-1], 1.0)
    if len(t_new) < 50:
        return None

    def interp(y):
        return interpolate.interp1d(
            time, y, kind="linear", fill_value="extrapolate"
        )(t_new)

    volt_r = interp(volt)
    curr_r = interp(curr)
    temp_r = interp(temp)
    ah_r   = interp(ah)

    # SOC from Ah field: Ah is cumulative Ah discharged (negative convention)
    # SOC = 1 - |Ah_discharged| / capacity
    ah_discharged = np.abs(ah_r - ah_r[0])
    soc = np.clip(1.0 - ah_discharged / capacity_ah, 0.0, 1.0)

    df = pd.DataFrame({
        "Time":        t_new,
        "Voltage":     volt_r,
        "Current":     curr_r,
        "Temperature": temp_r,
        "SOC":         soc,
    })

    df = df.dropna().reset_index(drop=True)
    if len(df) < 50:
        return None

    return df


def load_temperature(
    root: str,
    temperature: int,
    capacity_ah: float = 2.9,
) -> List[pd.DataFrame]:
    """Load all .mat files for a given temperature condition.
    
    Handles the Panasonic dataset's inconsistent nesting where 25degC
    is nested inside an extra 'Panasonic 18650PF Data' subdirectory.
    Searches both <root>/<temp>/ and <root>/*/<temp>/ automatically.
    """
    temp_dir = TEMP_DIR_MAP.get(temperature)
    if temp_dir is None:
        raise ValueError(
            f"Unknown temperature {temperature}C. Valid: {list(TEMP_DIR_MAP.keys())}"
        )

    root_path = Path(root)

    # Try direct path first: root/25degC/
    direct = root_path / temp_dir
    # Also try one level deeper: root/*/25degC/
    nested = list(root_path.glob(f"*/{temp_dir}"))

    candidates = []
    if direct.exists():
        candidates.append(direct)
    candidates.extend([p for p in nested if p.exists()])

    if not candidates:
        raise FileNotFoundError(
            f"Temperature folder '{temp_dir}' not found under: {root}\n"
            f"Searched: {direct} and {root_path}/*/{temp_dir}\n"
            f"Download from: https://data.mendeley.com/datasets/wykht8y7tg/1"
        )

    dfs = []
    for temp_path in candidates:
        all_files = sorted(
            glob.glob(str(temp_path / "**" / "*.mat"), recursive=True)
        )
        for fp in all_files:
            df = load_mat_file(fp, capacity_ah)
            if df is not None and len(df) > 10:
                df["source_file"] = os.path.basename(fp)
                df["temperature"] = temperature
                dfs.append(df)
                print(
                    f"    Loaded: {os.path.basename(fp)}"
                    f"  ({len(df):,} samples,"
                    f" SOC {df['SOC'].min():.2f}-{df['SOC'].max():.2f})"
                )

    return dfs


def load_panasonic(
    root: str,
    temperatures: List[int],
    capacity_ah: float = 2.9,
    **kwargs,  # absorb extra keyword args for compatibility
) -> Dict[int, List[pd.DataFrame]]:
    """
    Load Panasonic 18650PF dataset for specified temperatures.

    Args:
        root: Path to the folder that contains 25degC/, 10degC/, etc.
        temperatures: List of temperatures e.g. [25, 10, 0, -10]
        capacity_ah: Cell nominal capacity (default 2.9 Ah)

    Returns:
        Dict mapping temperature -> list of DataFrames
    """
    print(f"Loading Panasonic 18650PF from: {root}")
    result = {}
    for temp in temperatures:
        print(f"\n  Temperature: {temp}C")
        try:
            dfs = load_temperature(root, temp, capacity_ah)
            result[temp] = dfs
            print(f"  -> {len(dfs)} files loaded")
        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")
            result[temp] = []
    return result
