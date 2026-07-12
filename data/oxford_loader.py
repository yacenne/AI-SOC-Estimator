"""
Oxford Battery Degradation Dataset Loader

Source: https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac
Ref: C.R. Birkl, Oxford PhD Thesis, 2017

Cell specs: Kokam SLPB533459H4 pouch, 740 mAh, 8 cells (Cell1-Cell8), 40C constant.
Drive cycle: Urban Artemis.
Files: Cell1.mat ... Cell8.mat (MATLAB format).
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import interpolate


NOMINAL_CAPACITY_AH = 0.74


def _resample_series(time: np.ndarray, values: np.ndarray, target_hz: float) -> tuple:
    dt = 1.0 / target_hz
    t_new = np.arange(time[0], time[-1], dt)
    f = interpolate.interp1d(time, values, kind="linear", fill_value="extrapolate")
    return t_new, f(t_new)


def _coulomb_soc(current: np.ndarray, cap_ah: float, dt: float) -> np.ndarray:
    cap_as = cap_ah * 3600.0
    soc = np.empty(len(current))
    soc[0] = 1.0
    for i in range(1, len(current)):
        soc[i] = soc[i - 1] - current[i - 1] * dt / cap_as
    return np.clip(soc, 0.0, 1.0)


def load_oxford_cell(mat_path: str, target_hz: float = 1.0, nominal_capacity: float = NOMINAL_CAPACITY_AH) -> List[pd.DataFrame]:
    """
    Load a single Oxford cell .mat file.
    Returns list of DataFrames with columns: Time, Voltage, Current, Temperature, SOC.
    """
    try:
        mat = loadmat(mat_path, simplify_cells=True)
    except Exception as e:
        print(f"  [ERROR] {mat_path}: {e}")
        return []

    cell_id = Path(mat_path).stem
    data_key = next((k for k in mat if not k.startswith("__")), None)
    if data_key is None:
        return []

    raw = mat[data_key]
    dfs = []
    try:
        if isinstance(raw, np.ndarray) and raw.ndim == 2 and raw.shape[1] >= 4:
            data = raw[17:].astype(float)  # Skip header rows 1-17
            time = data[:, 0]
            voltage = data[:, 1]
            current = data[:, 2]
            temperature = data[:, 3]
            # Remove NaN rows
            mask = np.isfinite(time) & np.isfinite(voltage) & np.isfinite(current) & np.isfinite(temperature)
            time, voltage, current, temperature = time[mask], voltage[mask], current[mask], temperature[mask]
            if len(time) < 10:
                return []
            t_new, v_new = _resample_series(time, voltage, target_hz)
            _, i_new = _resample_series(time, current, target_hz)
            _, temp_new = _resample_series(time, temperature, target_hz)
            soc = _coulomb_soc(i_new, nominal_capacity, 1.0 / target_hz)
            df = pd.DataFrame({"Time": t_new, "Voltage": v_new, "Current": i_new, "Temperature": temp_new, "SOC": soc, "cell_id": cell_id})
            dfs.append(df)
    except Exception as e:
        print(f"  [WARN] Data extraction {mat_path}: {e}")
    print(f"  {cell_id}: {len(dfs)} segments loaded")
    return dfs


def load_oxford(root: str, n_cells: int = 8, target_hz: float = 1.0) -> Dict[str, List[pd.DataFrame]]:
    """
    Load all Oxford battery degradation cells.
    
    Args:
        root: Directory containing Cell1.mat ... CellN.mat.
        n_cells: Number of cells (default 8).
        target_hz: Target resampling rate.
    Returns:
        Dict: cell_id -> list of DataFrames.
    """
    print(f"Loading Oxford Battery Degradation from: {root}")
    result = {}
    for i in range(1, n_cells + 1):
        cid = f"Cell{i}"
        path = Path(root) / f"{cid}.mat"
        if not path.exists():
            print(f"  [WARN] {path} not found -- download from Oxford ORA repo")
            continue
        result[cid] = load_oxford_cell(str(path), target_hz)
    return result
