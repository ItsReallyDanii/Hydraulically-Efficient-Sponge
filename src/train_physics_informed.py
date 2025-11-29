"""
train_physics_informed.py

Physics-informed fine-tuning of the XylemAutoencoder using
differentiable proxy metrics for porosity and flow.

- Real-xylem images are used ONCE at startup to compute
  target proxy values (porosity_proxy, flow_proxy).
- During training, the autoencoder reconstructions are
  penalized if their proxy metrics deviate from those targets.

Evaluation with the full flow solver still happens separately
via `flow_simulation.py` + `analyze_flow_metrics.py`.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from src.model import XylemAutoencoder

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------

TARGET_SIZE = (256, 256)
DEVICE = torch.device("cpu")   # Camber/Jupyter CPU

REAL_XYLEM_DIRS = [
    "data/real_xylem_preprocessed",
    "data/real_xylem",
]


# -----------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------

def load_images_as_tensor(path: str) -> torch.Tensor:
    """
    Load and resize all grayscale images from a folder to consistent size.
    Returns [N,1,H,W] in [0,1].
    """
    imgs = []
    if not os.path.isdir(path):
        return None

    for f in sorted(os.listdir(path)):
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
            img = Image.open(os.path.join(path, f)).convert("L")
            img = img.resize(TARGET_SIZE, Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            imgs.append(torch.tensor(arr).unsqueeze(0))  # [1,H,W]

    if not imgs:
        return None

    return torch.stack(imgs)  # [N,1,H,W]


def compute_proxy_targets_from_real() -> tuple[float, float]:
    """
    Compute target proxy values (porosity_proxy, flow_proxy)
    from real-xylem images using only differentiable-style ops.

    Returns:
        porosity_target, flow_target (floats)
    """
    real_imgs = None
    for d in REAL_XYLEM_DIRS:
        real_imgs = load_images_as_tensor(d)
        if real_imgs is not None:
            print(f"📂 Loaded real xylem images from: {d}")
            break

    if real_imgs is None:
        # Fallback: reasonable defaults if real set is missing
        print("⚠️ No real xylem images found; using default proxy targets.")
        return 0.99, 0.01

    real_imgs = real_imgs.to(DEVICE)

    with torch.no_grad():
        # Porosity proxy: mean brightness
        porosity_proxy = real_imgs.mean()

        # Flow proxy: average gradient magnitude in x and y
        grad_y = real_imgs[:, :, 1:, :] - real_imgs[:, :, :-1, :]
        grad_x = real_imgs[:, :, :, 1:] - real_imgs[:, :, :, :-1]
        flow_proxy = (grad_x.abs().mean() + grad_y.abs().mean())

    porosity_target = float(porosity_proxy.item())
    flow_target = float(flow_proxy.item())

    print(f"🎯 Proxy targets from real xylem:")
    print(f"   Porosity_proxy_target ≈ {porosity_target:.6f}")
    print(f"   Flow_proxy_target     ≈ {flow_target:.6f}")

    return porosity_target, flow_target


def physics_proxy_loss(
    recon: torch.Tensor,
    porosity_target: float,
    flow_target: float,
):
    """
    Differentiable physics proxy loss computed directly on reconstructions.

    Args:
        recon: [N,1,H,W] tensor in [0,1]
        porosity_target: scalar target for mean brightness
        flow_target:     scalar target for gradient magnitude proxy

    Returns:
        phys_loss: torch scalar with gradient
        porosity_proxy_val: float (for logging)
    """
    # Porosity proxy: mean brightness
    porosity_proxy = recon.mean()

    # Flow proxy: gradient magnitude
    grad_y = recon[:, :, 1:, :] - recon[:, :, :-1, :]
    grad_x = recon[:, :, :, 1:] - recon[:, :, :, :-1]
    flow_proxy = (grad_x.abs().mean() + grad_y.abs().mean())

    eps = 1e-8
    p_loss = ((porosity_proxy - porosity_target) / (porosity_target + eps)) ** 2
    f_loss = ((flow_proxy - flow_target) / (flow_target + eps)) ** 2

    # Weights can be tuned; start symmetric
    w_p = 1.0
    w_f = 1.0

    phys_loss = w_p * p_loss + w_f * f_loss

    return phys_loss, float(porosity_proxy.item())


# -----------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------

def main():
    print("🌱 Physics-proxy fine-tuning started on", DEVICE)

    # 1) Load model
    model = XylemAutoencoder().to(DEVICE)
    model.load_state_dict(torch.load("results/model_hybrid.pth", map_location=DEVICE))
    model.train()

    # 2) Compute proxy targets from real xylem
    porosity_target, flow_target = compute_proxy_targets_from_real()

    # 3) Load synthetic training data
    data_path = "data/generated_microtubes"
    synth_imgs = load_images_as_tensor(data_path)
    if synth_imgs is None:
        raise RuntimeError(f"No synthetic images found in {data_path}")
    synth_imgs = synth_imgs.to(DEVICE)
    print(f"🧩 Loaded {len(synth_imgs)} generated structures → resized to {TARGET_SIZE}")

    recon_loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    logs = []

    for epoch in range(1, 101):
        optimizer.zero_grad()

        recon, _ = model(synth_imgs)
        recon_loss = recon_loss_fn(recon, synth_imgs)

        phys_loss, porosity_proxy_val = physics_proxy_loss(
            recon,
            porosity_target=porosity_target,
            flow_target=flow_target,
        )

        # Dynamic physics weighting (same schedule idea as before)
        weight_phys = 0.5 + 5 * (1 - np.exp(-epoch / 30.0))
        total_loss = recon_loss + weight_phys * phys_loss

        total_loss.backward()
        optimizer.step()

        # Gradient norm for logging
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item()

        logs.append({
            "epoch": epoch,
            "total": total_loss.item(),
            "recon": recon_loss.item(),
            "phys": phys_loss.item(),
            "PorosityProxy": porosity_proxy_val,
            "GradNorm": grad_norm,
        })

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/100 | "
                f"Total: {total_loss.item():.5f} | "
                f"Recon: {recon_loss.item():.5f} | "
                f"Phys: {phys_loss.item():.5f} | "
                f"PorosityProxy: {porosity_proxy_val:.5f} | "
                f"GradNorm: {grad_norm:.2e}"
            )

    # 4) Save results
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/model_physics_tuned.pth")
    pd.DataFrame(logs).to_csv("results/physics_training_log.csv", index=False)

    print("✅ Physics-proxy fine-tuning complete.")
    print("💾 Model saved → results/model_physics_tuned.pth")
    print("🧾 Training log saved → results/physics_training_log.csv")


if __name__ == "__main__":
    main()
