"""
train_surrogate.py

Train a CNN surrogate that maps xylem images → solver physics metrics:
    [Mean_K, Mean_dP/dy, FlowRate, Porosity, Anisotropy]

It consumes the dataset built by build_surrogate_dataset.py:
    results/surrogate_dataset.pt
        {
          "images":        FloatTensor [N, 1, 256, 256] in [0,1]
          "metrics":       FloatTensor [N, 5]
          "metric_names":  list[str] length 5
        }
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cpu")


# -------------------------------------------------------------------
# Surrogate model: small CNN → 5 physics metrics
# -------------------------------------------------------------------
class PhysicsSurrogateCNN(nn.Module):
    def __init__(self, n_outputs: int = 5):
        super().__init__()
        # Input images: [B, 1, 256, 256]
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 1x256x256 → 16x128x128

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 32x128x128 → 32x64x64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 64x64x64 → 64x32x32
        )

        self.head = nn.Sequential(
            nn.Flatten(),             # 64 * 32 * 32 = 65536
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return x


# -------------------------------------------------------------------
# Dataset loader
# -------------------------------------------------------------------
def load_surrogate_dataset(path: str = "results/surrogate_dataset.pt"):
    """
    Load the saved surrogate dataset:
        images: [N, 1, 256, 256]
        metrics: [N, 5]
        metric_names: list of 5 strings
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Surrogate dataset not found at {path}. "
            "Run build_surrogate_dataset.py first."
        )

    data = torch.load(path, map_location=DEVICE)
    images = data["images"].float().to(DEVICE)
    metrics = data["metrics"].float().to(DEVICE)
    metric_names = data.get("metric_names", ["Mean_K", "Mean_dP/dy", "FlowRate", "Porosity", "Anisotropy"])

    print("✅ Loaded surrogate dataset:")
    print(f"   images: {images.shape}")
    print(f"   metrics: {metrics.shape}")
    print(f"   metric_names: {metric_names}")

    return images, metrics, metric_names


# -------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------
def main():
    print("🧪 Training physics surrogate on", DEVICE)

    images, metrics, metric_names = load_surrogate_dataset()

    # Simple train/val split
    N = images.shape[0]
    n_train = int(0.8 * N)
    train_imgs = images[:n_train]
    train_y = metrics[:n_train]
    val_imgs = images[n_train:]
    val_y = metrics[n_train:]

    model = PhysicsSurrogateCNN(n_outputs=metrics.shape[1]).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 60
    best_val = float("inf")

    for epoch in range(1, num_epochs + 1):
        # ---- train ----
        model.train()
        optimizer.zero_grad()

        pred_train = model(train_imgs)
        loss_train = criterion(pred_train, train_y)
        loss_train.backward()
        optimizer.step()

        # ---- val ----
        model.eval()
        with torch.no_grad():
            pred_val = model(val_imgs)
            loss_val = criterion(pred_val, val_y)

        if loss_val.item() < best_val:
            best_val = loss_val.item()
            os.makedirs("results", exist_ok=True)
            torch.save(model.state_dict(), "results/physics_surrogate.pth")
            tag = "   ✅ New best val loss, model saved → results/physics_surrogate.pth"
        else:
            tag = ""

        print(
            f"Epoch {epoch:2d}/{num_epochs:2d} | "
            f"Train MSE: {loss_train.item():.6f} | "
            f"Val MSE: {loss_val.item():.6f}{tag}"
        )

    print("🏁 Surrogate training complete. Best val MSE:", best_val)


if __name__ == "__main__":
    main()
