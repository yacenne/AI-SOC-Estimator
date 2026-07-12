"""
CSFAT Fine-tuning Entry Point
==============================
Stage 2: Supervised fine-tuning for joint SOC estimation + fault detection.

Usage:
    python run_finetune.py [--config config/config.yaml] [--pretrain_ckpt checkpoints/pretrain_best.pt]
                          [--model csfat] [--run_name my_experiment]

Example:
    python run_finetune.py --model csfat --run_name csfat_panasonic
    python run_finetune.py --model vanilla_transformer --run_name baseline_transformer
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main(args):
    print("=" * 60)
    print("CSFAT Fine-tuning (Stage 2)")
    print("=" * 60)

    # Load config
    config = load_config(args.config)
    set_seed(config["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- 1. Load Data ----
    print("\n[1/5] Loading dataset...")
    from data.panasonic_loader import load_panasonic
    from data.dataset import build_windows_from_dfs, build_dataloaders

    pan_cfg = config["dataset"]["panasonic"]
    data = load_panasonic(
        root=pan_cfg["root"],
        temperatures=pan_cfg["train_temps"] + pan_cfg["val_temps"] + pan_cfg["test_temps"],
        capacity_ah=pan_cfg.get("capacity_ah", 2.9),
    )

    wc = config["window"]
    train_dfs = []
    for t in pan_cfg["train_temps"]:
        train_dfs.extend(data.get(t, []))
    val_dfs = []
    for t in pan_cfg["val_temps"]:
        val_dfs.extend(data.get(t, []))
    test_dfs = []
    for t in pan_cfg["test_temps"]:
        test_dfs.extend(data.get(t, []))

    train_w, train_l = build_windows_from_dfs(train_dfs, wc["size"], wc["stride"])
    val_w, val_l     = build_windows_from_dfs(val_dfs,   wc["size"], wc["stride"])
    test_w, test_l   = build_windows_from_dfs(test_dfs,  wc["size"], wc["stride"])

    print(f"  Train: {len(train_w):,} windows | Val: {len(val_w):,} | Test: {len(test_w):,}")

    if len(train_w) == 0:
        print("\n[ERROR] No training data found. Please check your dataset path:")
        print(f"  Expected: {pan_cfg['root']}")
        print("  Download from: https://data.mendeley.com/datasets/wykht8y7tg/1")
        sys.exit(1)

    # ---- 2. Setup Fault Injection ----
    print("\n[2/5] Setting up fault injection...")
    from faults.fault_injector import FaultInjector
    from faults.fault_schedule import CurriculumScheduler

    ft_cfg = config["finetune"]
    curr_cfg = ft_cfg["curriculum"]
    curriculum = CurriculumScheduler(
        warmup_epochs=curr_cfg["warmup_epochs"],
        ramp_epochs=curr_cfg["ramp_epochs"],
        max_fault_rate=curr_cfg["max_fault_rate"],
    )
    fault_injector = FaultInjector(
        fault_rate=0.0,  # Starts at 0; curriculum will update each epoch
        dual_sensor_rate=curr_cfg["dual_sensor_rate"],
        seed=config["project"]["seed"],
    )
    print(f"  Curriculum: warmup={curr_cfg['warmup_epochs']} | ramp={curr_cfg['ramp_epochs']} | max_rate={curr_cfg['max_fault_rate']}")

    # ---- 3. Build DataLoaders ----
    print("\n[3/5] Building DataLoaders...")
    train_loader, val_loader, test_loader, normalizer = build_dataloaders(
        train_w, train_l, val_w, val_l, test_w, test_l,
        batch_size=ft_cfg["batch_size"],
        fault_injector=fault_injector,
        return_fault_info=True,
    )

    # ---- 4. Build Model ----
    print(f"\n[4/5] Building model: {args.model}")
    from models.baselines import build_model
    model = build_model(args.model, config)
    print(f"  Parameters: {model.count_parameters():,}")

    # ---- 5. Train ----
    print("\n[5/5] Starting training...")
    from training.trainer import Trainer
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        checkpoint_dir=config["paths"]["checkpoints"],
        log_dir=config["paths"]["logs"],
    )

    # Load pretrained encoder (Stage 1) if provided
    if args.pretrain_ckpt and Path(args.pretrain_ckpt).exists():
        print(f"  Loading pretrained encoder from: {args.pretrain_ckpt}")
        trainer.load_pretrained_encoder(args.pretrain_ckpt)

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        fault_injector=fault_injector,
        curriculum_scheduler=curriculum,
        run_name=args.run_name,
    )

    print("\n[DONE] Fine-tuning complete.")
    print(f"  Best checkpoint: {config['paths']['checkpoints']}/{args.run_name}_best.pt")

    # Save history
    import json
    hist_path = Path(config["paths"]["logs"]) / f"{args.run_name}_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Training history: {hist_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSFAT Fine-tuning (Stage 2)")
    parser.add_argument("--config",        default="config/config.yaml",   help="Config file path")
    parser.add_argument("--model",         default="csfat",                help="Model: csfat | vanilla_transformer | lstm | bilstm")
    parser.add_argument("--run_name",      default="csfat_run1",           help="Experiment name (used for checkpoint naming)")
    parser.add_argument("--pretrain_ckpt", default=None,                   help="Path to Stage 1 pretrain checkpoint (optional)")
    args = parser.parse_args()
    main(args)
