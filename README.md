# CSFAT: Cross-Sensor Fault-Aware Transformer for Robust Battery SOC Estimation

> **Research project** implementing a novel Transformer-based model for EV battery State-of-Charge estimation that remains robust when BMS sensors are faulty.

## Project Overview

CSFAT treats each BMS sensor (Voltage, Current, Temperature) as an **independent token** in a Transformer. Cross-sensor attention learns inter-sensor compensation: when one sensor is faulty, the model automatically routes through the remaining healthy sensors.

### Key Contributions
1. **Novel architecture**: Cross-sensor sensor-as-token Transformer (iTransformer-inspired) for SOC
2. **Curriculum fault-injection training**: Model trained *with* faults, not just tested on them
3. **Systematic fault benchmark**: 6 fault types × 3 severities × 3 sensors (54 experiments)
4. **Joint model**: SOC estimation + fault detection in one forward pass

## Project Structure

```
EV-SOC-ESTIMATION/
├── config/
│   └── config.yaml              # All hyperparameters and dataset paths
├── data/
│   ├── panasonic_loader.py      # Panasonic 18650PF dataset loader
│   ├── nasa_loader.py           # NASA PCoE dataset loader
│   ├── oxford_loader.py         # Oxford Battery Degradation loader
│   └── dataset.py               # PyTorch Dataset + DataLoader utilities
├── faults/
│   ├── fault_types.py           # 6 fault type implementations + math models
│   ├── fault_injector.py        # Multi-sensor fault injector
│   └── fault_schedule.py        # Curriculum scheduling
├── models/
│   ├── sensor_embedding.py      # Per-sensor 1D Conv patch embedding
│   ├── cross_sensor_transformer.py  # Cross-sensor attention encoder
│   ├── csfat.py                 # Full CSFAT model
│   └── baselines.py             # VanillaTransformer, LSTM baselines
├── training/
│   ├── losses.py                # Huber + CrossEntropy + Attn Entropy losses
│   └── trainer.py               # Fine-tuning training loop
├── evaluation/
│   ├── metrics.py               # RMSE, MAE, Degradation Ratio, F1
│   └── evaluator.py             # Full 54-experiment robustness protocol
├── run_finetune.py              # Stage 2: supervised fine-tuning entry point
├── requirements.txt
└── README.md
```

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Datasets

**Panasonic 18650PF** (required, primary dataset):
```
URL: https://data.mendeley.com/datasets/wykht8y7tg/1
License: CC BY 4.0
Place in: data/raw/panasonic/
Structure: data/raw/panasonic/25degC/US06/*.csv, etc.
```

** ** (for cross-dataset generalization):
```
URL: https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip
Place in: data/raw/nasa/
Files needed: B0005.mat, B0006.mat, B0007.mat, B0018.mat
Also available on Kaggle as CSV
```

**Oxford Battery Degradation** (for aging validation):
```
URL: https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac
Place in: data/raw/oxford/
Files: Cell1.mat ... Cell8.mat
```

## Usage

### Fine-tuning CSFAT (Stage 2 — requires labeled SOC data)

```bash
# Train CSFAT with curriculum fault injection
python run_finetune.py --model csfat --run_name csfat_panasonic

# Train vanilla Transformer baseline (for comparison)
python run_finetune.py --model vanilla_transformer --run_name baseline_transformer

# Train BiLSTM baseline
python run_finetune.py --model bilstm --run_name baseline_bilstm
```

### Programmatic Usage (in notebooks or scripts)

```python
import yaml
from models.baselines import build_model
from faults.fault_types import FaultType, apply_fault
from faults.fault_injector import FaultInjector
import numpy as np

# Load config
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

# Build CSFAT model
model = build_model("csfat", config)
print(f"Parameters: {model.count_parameters():,}")

# Test fault injection
import torch
x = torch.randn(8, 128, 3)  # Batch of 8 windows, 128 timesteps, 3 sensors
out = model(x)
print("SOC predictions:", out["soc"].shape)  # (8, 1)
print("Fault logits:", out["fault_logits"].shape)  # (8, 3, 7)

# Test with sensor mask (simulate voltage sensor fault)
sensor_mask = torch.zeros(8, 3, dtype=torch.bool)
sensor_mask[:, 0] = True  # Mask voltage sensor
out_masked = model(x, sensor_mask)
print("SOC with masked V:", out_masked["soc"].shape)
```

## Fault Types

| Fault | Model | Physical Cause |
|-------|-------|----------------|
| **Dropout** | `x'(t) = 0` for `t ∈ [t_s, t_e]` | Open-circuit wiring |
| **Stuck-at** | `x'(t) = x(t_s)` for `t ≥ t_s` | Frozen ADC |
| **Drift** | `x'(t) = x(t) + α(t - t_s)` | Sensor aging |
| **Spike** | `x'(t_k) += A_k` at random times | EMI interference |
| **Bias** | `x'(t) = x(t) + b` | Calibration error |
| **Gain** | `x'(t) = g · x(t)` | ADC scaling error |

## Key Hyperparameters

See `config/config.yaml` for full configuration. Key defaults:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Window size | 128 steps | At 1 Hz = 128 seconds |
| Patch size | 16 steps | 8 patches per sensor |
| d_model | 64 | Embedding dimension |
| n_heads | 4 | Attention heads |
| n_layers | 3 | Transformer layers |
| Warmup epochs | 10 | Clean-data training |
| Max fault rate | 0.6 | 60% of windows faulted at peak |
| λ_fault | 0.3 | Fault loss weight |

## Datasets

| Dataset | Source | Use |
|---------|--------|-----|
| Panasonic 18650PF | [Mendeley](https://data.mendeley.com/datasets/wykht8y7tg/1) | Primary training/evaluation |
| NASA PCoE | [NASA](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) | Cross-dataset generalization |
| Oxford Degradation | [Oxford ORA](https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac) | Aging validation |

## References

- **iTransformer**: Liu et al., *ICLR 2024*. arXiv:2310.06625
- **PatchTST**: Nie et al., *ICLR 2023*. arXiv:2211.14730
- **TimeMAE**: arXiv:2301.03317
- **Panasonic 18650PF dataset**: Kollmeyer et al., Mendeley Data, 2020. DOI:10.17632/wykht8y7tg.1
