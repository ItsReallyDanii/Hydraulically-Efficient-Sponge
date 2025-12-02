"""
optimize_latent.py
------------------
Latent-space optimization + REAL solver check.

Pipeline:
  1. Load physics-tuned autoencoder + physics surrogate.
  2. Optimize a batch of latent codes z to match target physics
     according to the surrogate (K, flow, porosity, anisotropy).
  3. Decode the best z's to images and save them as PNGs.
  4. Run the TRUE solver (compute_flow_metrics) on each image.
  5. Save solver metrics to CSV and print their means.

This lets us directly compare:
  - surrogate-predicted metrics during optimization
  - real solver metrics after decoding.
"""

import os
import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image

from src.model import XylemAutoencoder
from src.train_surrogate import PhysicsSurrogateCNN
from src.flow_simulation_utils import compute_flow_metrics

DEVICE = "cpu"
LATENT_DIM = 32
BATCH_SIZE = 8
N_STEPS = 500
LR_Z = 0.05

# Real xylem physics targets (from your validation report)
K_TARGET = 0.24740
FLOW_TARGET = 0.00051
PORO_TARGET = 0.99042
ANISO_TARGET = 1.00870

OUT_DIR = "results/optimized_latent_v1"
os.makedirs(OUT_DIR, exist_ok=True)


def decode_to_numpy(ae, z_batch):
    """
    Decode a batch of latents → numpy images [B,H,W] in [0,1].
    """
    with torch.no_grad():
        x = ae.decode(z_batch)  # [B,1,256,256]
    x = x.cpu().numpy()
    x = np.clip(x, 0.0, 1.0)
    x = x[:, 0, :, :]  # drop channel
    return x


def save_images_png(images_np, out_dir, prefix="opt"):
    """
    Save batch of images (numpy [B,H,W] in [0,1]) to PNGs.
    Returns list of file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for idx, img in enumerate(images_np):
        img_uint8 = (img * 255.0).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(img_uint8, mode="L")
        fname = f"{prefix}_{idx:03d}.png"
        fpath = os.path.join(out_dir, fname)
        pil.save(fpath)
        paths.append(fpath)
    return paths


def main():
    print("🧪 Latent optimization + REAL solver evaluation started on cpu")

    # 1) Load autoencoder
    ae = XylemAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae_ckpt = "results/model_physics_tuned.pth"
    if not os.path.exists(ae_ckpt):
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {ae_ckpt}")
    ae.load_state_dict(torch.load(ae_ckpt, map_location=DEVICE))
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    # 2) Load surrogate
    surrogate = PhysicsSurrogateCNN().to(DEVICE)
    surrogate_ckpt = "results/physics_surrogate.pth"
    if not os.path.exists(surrogate_ckpt):
        raise FileNotFoundError(f"Surrogate checkpoint not found: {surrogate_ckpt}")
    surrogate.load_state_dict(torch.load(surrogate_ckpt, map_location=DEVICE))
    surrogate.eval()
    for p in surrogate.parameters():
        p.requires_grad_(False)

    print(
        f"🎯 Targets (surrogate-space): "
        f"K={K_TARGET:.5f}, Flow={FLOW_TARGET:.5f}, "
        f"Porosity={PORO_TARGET:.5f}, Aniso={ANISO_TARGET:.5f}"
    )

    # 3) Initialize latent codes
    z = torch.randn(BATCH_SIZE, LATENT_DIM, device=DEVICE, requires_grad=True)
    optimizer_z = torch.optim.Adam([z], lr=LR_Z)

    best_loss = math.inf
    best_z = None
    best_pred = None

    # 4) Optimize z using surrogate physics
    for step in range(1, N_STEPS + 1):
        optimizer_z.zero_grad()

        # Decode current latents → images
        x = ae.decode(z)  # [B,1,256,256]

        # Surrogate expects [B,1,H,W] float
        preds = surrogate(x)
        # Assume PhysicsSurrogateCNN outputs [B, 5]:
        # [mean_k, mean_dp_dy, flowrate, porosity, anisotropy]
        mean_k = preds[:, 0]
        flowrate = preds[:, 2]
        porosity = preds[:, 3]
        anisotropy = preds[:, 4]

        # Physics loss in surrogate-space
        loss_k = (mean_k - K_TARGET) ** 2
        loss_flow = (flowrate - FLOW_TARGET) ** 2
        loss_poro = (porosity - PORO_TARGET) ** 2
        loss_aniso = (anisotropy - ANISO_TARGET) ** 2

        # Weighting is somewhat arbitrary; we care most about porosity + K
        loss = (
            3.0 * loss_poro.mean()
            + 1.0 * loss_k.mean()
            + 0.5 * loss_flow.mean()
            + 0.5 * loss_aniso.mean()
        )

        loss.backward()
        optimizer_z.step()

        # Track best batch
        with torch.no_grad():
            cur_loss = loss.item()
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_z = z.detach().clone()
                best_pred = {
                    "K": mean_k.mean().item(),
                    "Flow": flowrate.mean().item(),
                    "Porosity": porosity.mean().item(),
                    "Aniso": anisotropy.mean().item(),
                }

        if step % 25 == 0 or step == 1:
            print(
                f"Step {step:4d}/{N_STEPS} | "
                f"Loss: {loss.item():.6f} | "
                f"Mean_K(surr): {mean_k.mean().item():.5f} | "
                f"Flow(surr): {flowrate.mean().item():.5f} | "
                f"Porosity(surr): {porosity.mean().item():.5f} | "
                f"Aniso(surr): {anisotropy.mean().item():.5f}"
            )

    print(f"✅ Finished optimization. Best surrogate loss: {best_loss:.6f}")
    if best_pred is not None:
        print(
            "   Best surrogate means → "
            f"K={best_pred['K']:.5f}, "
            f"Flow={best_pred['Flow']:.5f}, "
            f"Porosity={best_pred['Porosity']:.5f}, "
            f"Aniso={best_pred['Aniso']:.5f}"
        )

    # 5) Decode BEST latents and save PNGs
    print("🖼 Decoding best latents and saving PNGs...")
    best_imgs = decode_to_numpy(ae, best_z.to(DEVICE))
    img_paths = save_images_png(best_imgs, OUT_DIR, prefix="opt")
    print(f"   Saved {len(img_paths)} images to {OUT_DIR}")

    # 6) Run TRUE solver on each image
    print("🌊 Running TRUE solver (compute_flow_metrics) on optimized images...")
    records = []
    for fpath, img in zip(img_paths, best_imgs):
        metrics = compute_flow_metrics(img)  # img is [H,W] in [0,1]
        # Expect keys: mean_k, mean_dp/dy, flowrate, porosity, anisotropy
        records.append(
            {
                "file": os.path.basename(fpath),
                "mean_k": float(metrics.get("mean_k", 0.0)),
                "mean_dp/dy": float(metrics.get("mean_dp/dy", 0.0)),
                "flowrate": float(metrics.get("flowrate", 0.0)),
                "porosity": float(metrics.get("porosity", 0.0)),
                "anisotropy": float(metrics.get("anisotropy", 0.0)),
            }
        )

    df = pd.DataFrame(records)
    csv_path = os.path.join(OUT_DIR, "optimized_solver_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"💾 Saved solver metrics → {csv_path}")

    # 7) Print mean solver metrics
    mean_k = df["mean_k"].mean()
    mean_dpdy = df["mean_dp/dy"].mean()
    mean_flow = df["flowrate"].mean()
    mean_poro = df["porosity"].mean()
    mean_aniso = df["anisotropy"].mean()

    print("\n📊 TRUE solver metrics over optimized batch:")
    print(f"   mean_k      (solver) = {mean_k:.5f}")
    print(f"   mean_dp/dy  (solver) = {mean_dpdy:.5f}")
    print(f"   flowrate    (solver) = {mean_flow:.5f}")
    print(f"   porosity    (solver) = {mean_poro:.5f}")
    print(f"   anisotropy  (solver) = {mean_aniso:.5f}")

    print("\n🔍 Now compare these to:")
    print("   Real xylem:")
    print(f"     K      ≈ {K_TARGET:.5f}")
    print(f"     Flow   ≈ {FLOW_TARGET:.5f}")
    print(f"     Poros. ≈ {PORO_TARGET:.5f}")
    print(f"     Aniso  ≈ {ANISO_TARGET:.5f}")


if __name__ == "__main__":
    main()
