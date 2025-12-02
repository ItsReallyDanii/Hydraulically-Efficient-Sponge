"""
train_physics_informed_v03.py

Xylem v0.3:
- Uses the trained physics surrogate to match BOTH mean and variance
  of (K, dP/dy, FlowRate, Porosity, Anisotropy) to the REAL xylem
  distribution.

This leaves v0.2 intact and gives you a clearly labeled v0.3 experiment.
"""

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from PIL import Image

from src.model import XylemAutoencoder
from src.train_surrogate import PhysicsSurrogateCNN  # uses the same class as when you trained the surrogate

DEVICE = torch.device("cpu")
TARGET_SIZE = (256, 256)

REAL_METRICS_CSV = "results/flow_metrics/flow_metrics.csv"
BASE_MODEL_PATH = "results/model_hybrid.pth"          # you can change to model_physics_informed.pth if you want
SURROGATE_PATH   = "results/physics_surrogate.pth"
OUT_MODEL_PATH   = "results/model_physics_v03.pth"
OUT_LOG_CSV      = "results/physics_training_log_v03.csv"


# -----------------------------
# Data loading
# -----------------------------
def load_and_preprocess_images(path):
    """Load and resize all grayscale images from a folder to consistent size."""
    imgs = []
    for f in sorted(os.listdir(path)):
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
            img = Image.open(os.path.join(path, f)).convert("L")
            img = img.resize(TARGET_SIZE, Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            imgs.append(torch.tensor(arr).unsqueeze(0))  # [1,H,W]
    if not imgs:
        raise RuntimeError(f"No images found in {path}")
    batch = torch.stack(imgs)  # [N,1,H,W]
    return batch.to(DEVICE)


# -----------------------------
# Real-xylem target stats
# -----------------------------
METRIC_COLUMNS = ["Mean_K", "Mean_dP/dy", "FlowRate", "Porosity", "Anisotropy"]

def load_real_targets(csv_path=REAL_METRICS_CSV):
    """
    Load real-xylem flow metrics and compute per-metric mean & variance.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Real metrics CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # If there's a Type column, filter to real
    if "Type" in df.columns:
        real_df = df[df["Type"].str.lower() == "real"].copy()
        if real_df.empty:
            raise ValueError("No 'Real' rows found in flow_metrics.csv")
    else:
        real_df = df

    missing = [c for c in METRIC_COLUMNS if c not in real_df.columns]
    if missing:
        raise ValueError(f"Missing columns in real metrics CSV: {missing}")

    means = real_df[METRIC_COLUMNS].mean().values.astype(np.float32)
    vars_ = real_df[METRIC_COLUMNS].var(ddof=0).values.astype(np.float32)  # population variance

    return torch.tensor(means, device=DEVICE), torch.tensor(vars_, device=DEVICE)


# -----------------------------
# Training
# -----------------------------
def main():
    print("🌱 Surrogate-based distributional physics fine-tuning (v0.3) started on", DEVICE)

    # 1) Load real-xylem targets
    real_means, real_vars = load_real_targets()
    eps = 1e-6
    print("🎯 Real-physics targets from solver (means):")
    for name, val in zip(METRIC_COLUMNS, real_means.tolist()):
        print(f"   {name:<10} ≈ {val:.6f}")
    print("🎯 Real-physics targets from solver (vars):")
    for name, val in zip(METRIC_COLUMNS, real_vars.tolist()):
        print(f"   {name:<10} ≈ {val:.6f}")

    # 2) Load model
    model = XylemAutoencoder().to(DEVICE)
    if not os.path.exists(BASE_MODEL_PATH):
        raise FileNotFoundError(f"Base model not found at {BASE_MODEL_PATH}")
    model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=DEVICE))
    model.train()

    # 3) Load surrogate
    surrogate = PhysicsSurrogateCNN().to(DEVICE)
    if not os.path.exists(SURROGATE_PATH):
        raise FileNotFoundError(f"Surrogate model not found at {SURROGATE_PATH}")
    surrogate.load_state_dict(torch.load(SURROGATE_PATH, map_location=DEVICE))
    surrogate.eval()  # frozen

    # 4) Load generated structures (same as v0.2)
    data_path = "data/generated_microtubes"
    imgs = load_and_preprocess_images(data_path)
    print(f"🧩 Loaded {imgs.shape[0]} generated structures → resized to {TARGET_SIZE}")

    recon_loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    logs = []
    num_epochs = 100

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()

        recon, _ = model(imgs)
        recon_loss = recon_loss_fn(recon, imgs)

        # ---------- physics via surrogate ----------
        with torch.no_grad():
            # surrogate outputs 5 metrics per image
            preds = surrogate(recon).view(-1, len(METRIC_COLUMNS))  # [N,5]

        batch_means = preds.mean(dim=0)          # [5]
        batch_vars  = preds.var(dim=0, unbiased=False)  # [5]

        # mean + variance matching, normalised by real variance
        mean_term = ((batch_means - real_means) ** 2) / (real_vars + eps)
        var_term  = ((batch_vars  - real_vars)  ** 2) / ((real_vars + eps) ** 2)

        # weight variance lower than mean to start
        alpha_var = 0.3
        phys_loss_per_metric = mean_term + alpha_var * var_term
        phys_loss = phys_loss_per_metric.sum()

        # λ_phys schedule (starts modest, ramps up)
        lambda_phys = 0.5 + 5.0 * (1.0 - math.exp(-epoch / 30.0))

        total_loss = recon_loss + lambda_phys * phys_loss
        total_loss.backward()
        optimizer.step()

        # quick logging of some metrics
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5

        log_row = {
            "epoch": epoch,
            "total": total_loss.item(),
            "recon": recon_loss.item(),
            "phys": phys_loss.item(),
            "lambda_phys": lambda_phys,
            "Mean_K": batch_means[0].item(),
            "Mean_dP_dy": batch_means[1].item(),
            "FlowRate": batch_means[2].item(),
            "Porosity": batch_means[3].item(),
            "Anisotropy": batch_means[4].item(),
            "Var_K": batch_vars[0].item(),
            "Var_dP_dy": batch_vars[1].item(),
            "Var_FlowRate": batch_vars[2].item(),
            "Var_Porosity": batch_vars[3].item(),
            "Var_Anisotropy": batch_vars[4].item(),
            "GradNorm": grad_norm,
        }
        logs.append(log_row)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Total: {total_loss.item():.5f} | "
                f"Recon: {recon_loss.item():.5f} | "
                f"Phys: {phys_loss.item():.5f} | "
                f"λ_phys: {lambda_phys:.2f} | "
                f"Porosity_mean: {batch_means[3].item():.5f}"
            )

    # save
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), OUT_MODEL_PATH)
    pd.DataFrame(logs).to_csv(OUT_LOG_CSV, index=False)

    print("✅ v0.3 physics fine-tuning complete.")
    print(f"💾 Model saved → {OUT_MODEL_PATH}")
    print(f"🧾 Training log saved → {OUT_LOG_CSV}")


if __name__ == "__main__":
    main()
