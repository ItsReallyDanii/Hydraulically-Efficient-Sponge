"""
train_surrogate.py
------------------
Train a small CNN surrogate to approximate the flow solver:

    image (1x256x256) → [Mean_K, Mean_dP/dy, FlowRate, Porosity, Anisotropy]

The trained weights are saved as:
    results/physics_surrogate.pth

`PhysicsSurrogateCNN` is defined at module level so it can be imported from
other scripts, e.g.:

    from src.train_surrogate import PhysicsSurrogateCNN
"""

import os
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
#  Model definition — this is what train_physics_informed_v03
#  expects to import as the surrogate.
# ============================================================

class PhysicsSurrogateCNN(nn.Module):
    """
    Simple convolutional regressor:
        1x256x256 → conv stack → flatten → FC → 5 metrics
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),  # 256 → 128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                       # 128 → 64

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # 64 → 64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                       # 64 → 32

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 32 → 32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                       # 32 → 16
        )

        # 64 channels * 16 * 16 = 16384 features
        flat_dim = 64 * 16 * 16

        self.head = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 5)   # [Mean_K, Mean_dP/dy, FlowRate, Porosity, Anisotropy]
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


# ============================================================
#  Training script
# ============================================================

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧪 Training physics surrogate on {DEVICE}")

    dataset_path = "results/surrogate_dataset.pt"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"{dataset_path} not found. Run build_surrogate_dataset.py first."
        )

    data = torch.load(dataset_path, map_location=DEVICE)
    images = data["images"].to(DEVICE)          # [N, 1, 256, 256]
    metrics = data["metrics"].to(DEVICE)        # [N, 5]
    metric_names = data.get("metric_names", ["Mean_K", "Mean_dP/dy", "FlowRate", "Porosity", "Anisotropy"])

    print("✅ Loaded surrogate dataset:")
    print(f"   images: {images.shape}")
    print(f"   metrics: {metrics.shape}")
    print(f"   metric_names: {metric_names}")

    # ----- train/val split -----
    N = images.size(0)
    n_val = max(1, int(0.2 * N))
    n_train = N - n_val

    train_imgs, val_imgs = torch.split(images, [n_train, n_val])
    train_y,   val_y   = torch.split(metrics, [n_train, n_val])

    train_loader = DataLoader(
        TensorDataset(train_imgs, train_y),
        batch_size=16,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_imgs, val_y),
        batch_size=16,
        shuffle=False
    )

    model = PhysicsSurrogateCNN().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs("results", exist_ok=True)
    best_val = float("inf")

    EPOCHS = 60
    for epoch in range(1, EPOCHS + 1):
        # ----- train -----
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        train_loss /= n_train

        # ----- validate -----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item() * x.size(0)

        val_loss /= n_val

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "results/physics_surrogate.pth")
            print(
                f"Epoch {epoch:2d}/{EPOCHS} | "
                f"Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}\n"
                f"   ✅ New best val loss, model saved → results/physics_surrogate.pth"
            )
        else:
            print(
                f"Epoch {epoch:2d}/{EPOCHS} | "
                f"Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}"
            )

    print(f"🏁 Surrogate training complete. Best val MSE: {best_val:.6f}")


if __name__ == "__main__":
    main()
