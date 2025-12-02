import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from PIL import Image

from src.model import XylemAutoencoder
from src.train_surrogate import PhysicsSurrogateCNN  # same class you used to train the surrogate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (256, 256)


# ----------------------------
# Data loading
# ----------------------------
def load_and_preprocess_images(path):
    """Load all images from a folder, convert to grayscale, resize, normalize to [0,1]."""
    imgs = []
    for fname in sorted(os.listdir(path)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
            p = os.path.join(path, fname)
            img = Image.open(p).convert("L")
            img = img.resize(IMG_SIZE, Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0  # [H,W] in [0,1]
            imgs.append(torch.from_numpy(arr).unsqueeze(0))  # [1,H,W]
    if not imgs:
        raise RuntimeError(f"No images found in {path}")
    stack = torch.stack(imgs)  # [N,1,H,W]
    return stack.to(DEVICE)


# ----------------------------
# Helper: real-physics targets
# ----------------------------
def load_real_solver_stats(csv_path="results/flow_metrics/flow_metrics.csv"):
    """
    Load solver metrics and compute REAL-xylem means and variances.
    Expects columns: ['Mean_K','Mean_dP/dy','FlowRate','Porosity','Anisotropy','Type']
    """
    df = pd.read_csv(csv_path)
    if "Type" in df.columns:
        real = df[df["Type"].str.lower() == "real"].copy()
    else:
        real = df.copy()

    means = {
        "Mean_K": float(real["Mean_K"].mean()),
        "Mean_dP/dy": float(real["Mean_dP/dy"].mean()),
        "FlowRate": float(real["FlowRate"].mean()),
        "Porosity": float(real["Porosity"].mean()),
        "Anisotropy": float(real["Anisotropy"].mean()),
    }
    vars_ = {
        "Mean_K": float(real["Mean_K"].var(ddof=0) + 1e-12),
        "Mean_dP/dy": float(real["Mean_dP/dy"].var(ddof=0) + 1e-12),
        "FlowRate": float(real["FlowRate"].var(ddof=0) + 1e-12),
        "Porosity": float(real["Porosity"].var(ddof=0) + 1e-12),
        "Anisotropy": float(real["Anisotropy"].var(ddof=0) + 1e-12),
    }
    return means, vars_


# ----------------------------
# Porosity proxy from image
# ----------------------------
def porosity_proxy_from_image(batch):
    """
    Simple differentiable porosity proxy.

    Assumption: pores are darker than background → high porosity ≈ lots of dark pixels.
    So we use: porosity_proxy = mean(1 - intensity).

    If this turns out inverted when you look at images, we can flip it later.
    """
    # batch: [N,1,H,W] in [0,1]
    return (1.0 - batch).mean(dim=[1, 2, 3])  # [N]


# ----------------------------
# Training loop (v0.4)
# ----------------------------
def main():
    print("🌿 Surrogate-based *porosity-aware* physics fine-tuning (v0.4) started on", DEVICE)

    # 1) Load real solver stats
    real_means, real_vars = load_real_solver_stats("results/flow_metrics/flow_metrics.csv")
    print("🎯 Real-physics targets from solver (means):")
    for k, v in real_means.items():
        print(f"   {k:<10} ≈ {v:.6f}")
    print("🎯 Real-physics targets from solver (vars):")
    for k, v in real_vars.items():
        print(f"   {k:<10} ≈ {v:.6f}")

    # 2) Load model + surrogate
    model = XylemAutoencoder().to(DEVICE)
    model.load_state_dict(torch.load("results/model_hybrid.pth", map_location=DEVICE))
    model.train()

    surrogate = PhysicsSurrogateCNN().to(DEVICE)
    surrogate.load_state_dict(torch.load("results/physics_surrogate.pth", map_location=DEVICE))
    surrogate.eval()  # we never train surrogate here

    # 3) Load synthetic images to fine-tune on
    data_path = "data/generated_microtubes"
    imgs = load_and_preprocess_images(data_path)
    print(f"🧩 Loaded {imgs.shape[0]} generated structures → resized to {IMG_SIZE}")

    # 4) Optimizer / loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    recon_loss_fn = nn.MSELoss()

    num_epochs = 100
    logs = []

    # handy scalars
    K_target = real_means["Mean_K"]
    dP_target = real_means["Mean_dP/dy"]
    Flow_target = real_means["FlowRate"]
    Por_target = real_means["Porosity"]
    Aniso_target = real_means["Anisotropy"]

    # weights for each metric inside physics loss (all in normalized space)
    w_K = 1.0
    w_dP = 0.5
    w_Flow = 1.0
    w_Aniso = 0.2
    w_Por_proxy = 2.0  # we care a lot about porosity looking like real

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()

        # Forward pass through AE
        recon, _ = model(imgs)
        recon_loss = recon_loss_fn(recon, imgs)

        # Physics surrogate predictions (no grad through surrogate weights)
        with torch.no_grad():
            surrogate_out = surrogate(recon)  # [N,5]

        # Mean over batch
        K_pred_mean = surrogate_out[:, 0].mean()
        dP_pred_mean = surrogate_out[:, 1].mean()
        Flow_pred_mean = surrogate_out[:, 2].mean()
        Por_pred_mean = surrogate_out[:, 3].mean()
        Aniso_pred_mean = surrogate_out[:, 4].mean()

        # Image-based porosity proxy (differentiable)
        por_proxy_batch = porosity_proxy_from_image(recon)  # [N]
        por_proxy_mean = por_proxy_batch.mean()

        # Normalized squared errors (divide by target^2 to put things in comparable scale)
        def norm_sq(err, target):
            return (err ** 2) / (target ** 2 + 1e-12)

        loss_K = norm_sq(K_pred_mean - K_target, K_target)
        loss_dP = norm_sq(dP_pred_mean - dP_target, max(abs(dP_target), 1e-4))
        loss_Flow = norm_sq(Flow_pred_mean - Flow_target, Flow_target)
        loss_Aniso = norm_sq(Aniso_pred_mean - Aniso_target, Aniso_target)
        loss_Por_proxy = norm_sq(por_proxy_mean - Por_target, Por_target)

        # Total physics loss (inside-batch)
        phys_loss = (
            w_K * loss_K
            + w_dP * loss_dP
            + w_Flow * loss_Flow
            + w_Aniso * loss_Aniso
            + w_Por_proxy * loss_Por_proxy
        )

        # Epoch-dependent physics weight λ_phys
        # Start gentle, grow but not to insane values
        lambda_phys = 0.5 + 4.5 * (1.0 - np.exp(-epoch / 30.0))

        total_loss = recon_loss + lambda_phys * phys_loss
        total_loss.backward()
        optimizer.step()

        # simple grad norm for sanity
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5

        logs.append(
            {
                "epoch": epoch,
                "total": float(total_loss.item()),
                "recon": float(recon_loss.item()),
                "phys": float(phys_loss.item()),
                "lambda_phys": float(lambda_phys),
                "K_pred_mean": float(K_pred_mean.item()),
                "Flow_pred_mean": float(Flow_pred_mean.item()),
                "Por_pred_mean": float(Por_pred_mean.item()),
                "Por_proxy_mean": float(por_proxy_mean.item()),
                "GradNorm": float(grad_norm),
            }
        )

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Total: {total_loss.item():.5f} | "
                f"Recon: {recon_loss.item():.5f} | "
                f"Phys: {phys_loss.item():.5f} | "
                f"λ_phys: {lambda_phys:.2f} | "
                f"K_mean: {K_pred_mean.item():.5f} | "
                f"Flow_mean: {Flow_pred_mean.item():.5f} | "
                f"Por_solver_mean: {Por_pred_mean.item():.5f} | "
                f"Por_proxy_mean: {por_proxy_mean.item():.5f} | "
                f"GradNorm: {grad_norm:.2e}"
            )

    # Save model + logs
    os.makedirs("results/physics_informed_training", exist_ok=True)
    out_model_path = "results/model_physics_v04.pth"
    torch.save(model.state_dict(), out_model_path)
    pd.DataFrame(logs).to_csv("results/physics_training_log_v04.csv", index=False)

    print("✅ Physics fine-tuning v0.4 complete.")
    print(f"💾 Model saved → {out_model_path}")
    print("🧾 Training log saved → results/physics_training_log_v04.csv")


if __name__ == "__main__":
    main()
