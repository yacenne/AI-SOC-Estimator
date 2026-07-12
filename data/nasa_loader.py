"""
NASA PCoE Battery Dataset Loader

Download:
  https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip
  Or: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

Expected:
  data/raw/nasa/B0005.mat, B0006.mat, B0007.mat, B0018.mat

Use scipy.io.loadmat with simplify_cells=True to parse the nested MATLAB structs.
Each cell has charge/discharge/impedance cycles.
Sensor channels: Voltage_measured, Current_measured, Temperature_measured, Time.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import interpolate


NOMINAL_CAPACITY_AH = 2.0


def _resample(time: np.ndarray, values: np.ndarray, target_hz: float = 1.0) -> tuple:
    dt = 1.0 / target_hz
    t_new = np.arange(time[0], time[-1], dt)
    f = interpolate.interp1d(time, values, kind="linear", fill_value="extrapolate")
    return t_new, f(t_new)


def _coulomb_soc(current: np.ndarray, capacity_ah: float, dt: float = 1.0) -> np.ndarray:
    """Coulomb counting; NASA convention: negative current = discharge."""
    cap_as = capacity_ah * 3600.0
    soc = np.empty(len(current))
    soc[0] = 1.0
    for i in range(1, len(current)):
        # Flip sign: negative I = discharge, so subtract negative = add charge out
        soc[i] = soc[i - 1] + (current[i - 1] * dt) / cap_as  # negative I -> decreasing SOC
    return np.clip(soc, 0.0, 1.0)


def load_nasa_cell(mat_path: str, cycle_type: str = "discharge", target_hz: float = 1.0, nominal_capacity: float = NOMINAL_CAPACITY_AH) -> List[pd.DataFrame]:
    """
    Load discharge cycles from a NASA PCoE .mat file.
    Returns list of DataFrames with columns: Time, Voltage, Current, Temperature, SOC, cycle_number, cell_id.
    """
    try:
        mat = loadmat(mat_path, simplify_cells=True)
    except Exception as e:
        print(f"[ERROR] Cannot load {mat_path}: {e}")
        return []

    cell_id = Path(mat_path).stem
    root_key = [k for k in mat if not k.startswith("__")][0]
    try:
        cycles_raw = mat[root_key]["cycle"]
    except Exception as e:
        print(f"[ERROR] Cannot find cycles in {mat_path}: {e}")
        return []

    dfs = []
    cycle_num = 0
    for cyc in cycles_raw:
        try:
            if cyc.get("type", "") != cycle_type:
                continue
            cycle_num += 1
            d = cyc["data"]
            voltage = np.array(d["Voltage_measured"]).flatten()
            current = np.array(d["Current_measured"]).flatten()
            temperature = np.array(d["Temperature_measured"]).flatten()
            time = np.array(d["Time"]).flatten()
            n = min(len(voltage), len(current), len(temperature), len(time))
            if n < 10:
                continue
            t_new, v_new = _resample(time[:n], voltage[:n], target_hz)
            _, i_new = _resample(time[:n], current[:n], target_hz)
            _, temp_new = _resample(time[:n], temperature[:n], target_hz)
            soc = _coulomb_soc(i_new, nominal_capacity, 1.0 / target_hz)
            df = pd.DataFrame({"Time": t_new, "Voltage": v_new, "Current": i_new, "Temperature": temp_new, "SOC": soc})
            df["cycle_number"] = cycle_num
            df["cell_id"] = cell_id
            dfs.append(df)
        except Exception as e:
            print(f"  [WARN] Cycle {cycle_num} in {cell_id}: {e}")
    print(f"  {cell_id}: {len(dfs)} {cycle_type} cycles")
    return dfs


def load_nasa(root: str, cell_ids: Optional[List[str]] = None, cycle_type: str = "discharge", target_hz: float = 1.0) -> Dict[str, List[pd.DataFrame]]:
    """
    Load NASA PCoE battery dataset.
    
    Args:
        root: Directory containing B00XX.mat files.
        cell_ids: Cell IDs to load. Defaults to ["B0005", "B0006", "B0007", "B0018"].
        cycle_type: 'discharge' or 'charge'.
        target_hz: Resample frequency.
    Returns:
        Dict: cell_id -> list of cycle DataFrames.
    """
    if cell_ids is None:
        cell_ids = ["B0005", "B0006", "B0007", "B0018"]
    print(f"Loading NASA PCoE from: {root}")
    result = {}
    for cid in cell_ids:
        path = Path(root) / f"{cid}.mat"
        if not path.exists():
            print(f"  [WARN] {path} not found -- download from NASA PCoE repo")
            continue
        result[cid] = load_nasa_cell(str(path), cycle_type, target_hz)
    return result
