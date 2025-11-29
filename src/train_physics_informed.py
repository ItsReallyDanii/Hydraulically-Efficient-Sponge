import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from PIL import Image
from src.model import XylemAutoencoder
from src.flow_simulation_utils import compute_flow_metrics

TARGET_SIZE = (256, 256)

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
    return torch.stack(imgs)  # [N,1,H,W]

def safe_get(metrics, *possible_keys, default=0.0):
    """Try multiple keys to safely extract a metric from the dict."""
    for k in possible_keys:
        if k in metrics and metrics[k] is not None:
            return float(metrics[k])
    return default

def compute_physics_loss(batch_imgs):
    """Compute permeability + porosity regularization losses."""
    K_list, P_list = [], []
    for img_tensor in batch_imgs:
        img_np = img_tensor.detach().cpu().squeeze().numpy()
        metrics = compute_flow_metrics(img_np)

        K_val = safe_get(metrics, "Mean_K", "mean_k", "K", default=0.0)
        P_val = safe_get(metrics, "Porosity", "porosity", default=0.0)

        if np.isnan(K_val) or np.isnan(P_val):
            K_val, P_val = 0.0, 0.0

        K_list.append(K_val)
        P_list.append(P_val)

    K_mean, P_mean = np.mean(K_list), np.mean(P_list)

    # Target values for realistic xylem flow
    K_target, P_target = 0.5, 0.9
    K_loss = (K_mean - K_target) ** 2
    P_loss = (P_mean - P_target) ** 2

    phys_loss = K_loss + P_loss
    return phys_loss, K_mean, P_mean

def main():
    print("🌱 Physics-informed fine-tuning started on cpu")

    # Load model
    model = XylemAutoencoder()
    model.load_state_dict(torch.load("results/model_hybrid.pth", map_location="cpu"))
    model.train()

    # Load data
    data_path = "data/generated_microtubes"
    imgs = load_and_preprocess_images(data_path)
    print(f"🧩 Loaded {len(imgs)} generated structures → resized to {TARGET_SIZE}")

    recon_loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    logs = []
    for epoch in range(1, 101):
        optimizer.zero_grad()
        recon, _ = model(imgs)
        recon_loss = recon_loss_fn(recon, imgs)

        phys_loss, K_mean, P_mean = compute_physics_loss(recon)

        # Dynamic physics weighting
        weight_phys = 0.5 + 5 * (1 - np.exp(-epoch / 30))
        total_loss = recon_loss + weight_phys * phys_loss

        total_loss.backward()
        optimizer.step()

        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)

        logs.append({
            "epoch": epoch,
            "total": total_loss.item(),
            "recon": recon_loss.item(),
            "phys": phys_loss.item(),
            "K": K_mean,
            "Porosity": P_mean,
            "GradNorm": grad_norm
        })

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/100 | Total: {total_loss.item():.5f} | Recon: {recon_loss.item():.5f} | "
                  f"Phys: {phys_loss.item():.5f} | K: {K_mean:.5f} | Porosity: {P_mean:.5f} | GradNorm: {grad_norm:.2e}")

    # Save results
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/model_physics_tuned.pth")
    pd.DataFrame(logs).to_csv("results/physics_training_log.csv", index=False)
    print("✅ Physics-informed fine-tuning complete.")
    print("💾 Model saved → results/model_physics_tuned.pth")
    print("🧾 Training log saved → results/physics_training_log.csv")

if __name__ == "__main__":
    main()
