import torch
import yaml
import numpy as np
import argparse
from models.baselines import build_model
from data.panasonic_loader import load_panasonic
from data.dataset import build_windows_from_dfs, SensorNormalizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="csfat", help="Model name: csfat | lstm | vanilla_transformer")
    parser.add_argument("--ckpt", default="checkpoints/csfat_clean_final_best.pt", help="Path to checkpoint")
    args = parser.parse_args()

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    pan_cfg = config["dataset"]["panasonic"]
    print("Loading Panasonic dataset...")
    data = load_panasonic(root=pan_cfg["root"],
                           temperatures=pan_cfg["train_temps"] + pan_cfg["val_temps"],
                           capacity_ah=pan_cfg.get("capacity_ah", 2.9))

    # Rebuild normalizer from training set
    print("Fitting normalizer on training set...")
    train_dfs = []
    for t in pan_cfg["train_temps"]:
        train_dfs.extend(data.get(t, []))
    wc = config["window"]
    train_w, _ = build_windows_from_dfs(train_dfs, wc["size"], wc["stride"])
    
    normalizer = SensorNormalizer()
    normalizer.fit(train_w)

    # Load validation windows
    val_dfs = []
    for t in pan_cfg["val_temps"]:
        val_dfs.extend(data.get(t, []))
    val_w, val_l = build_windows_from_dfs(val_dfs, wc["size"], wc["stride"])
    
    # Normalize validation windows
    val_w_norm = normalizer.transform(val_w)

    # Load model
    print(f"Building model '{args.model}' and loading checkpoint from '{args.ckpt}'...")
    model = build_model(args.model, config)
    
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    
    # Check if checkpoint needs RevIN (backwards compatibility for old checkpoints)
    has_revin = any("revin" in k for k in state_dict.keys())
    if has_revin:
        from models.csfat import RevIN
        model.revin = RevIN(config["model"]["n_sensors"])
        
    model.load_state_dict(state_dict)
    model.eval()

    # Predict
    X = torch.tensor(val_w_norm, dtype=torch.float32)
    y_true = val_l
    with torch.no_grad():
        y_pred = model(X)["soc"].squeeze().numpy()

    rmse = np.sqrt(((y_true - y_pred)**2).mean())
    mae = np.mean(np.abs(y_true - y_pred))
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Validation temperature: {pan_cfg['val_temps']}")
    print(f"Validation samples:     {len(y_true):,}")
    print(f"True SOC Range:         {y_true.min():.4f} to {y_true.max():.4f}")
    print(f"Pred SOC Range:         {y_pred.min():.4f} to {y_pred.max():.4f}")
    print(f"Guessing Mean RMSE:     {np.sqrt(((y_true - y_true.mean())**2).mean())*100:.3f}%")
    print(f"Model RMSE:             {rmse*100:.3f}%")
    print(f"Model MAE:              {mae*100:.3f}%")

if __name__ == "__main__":
    main()