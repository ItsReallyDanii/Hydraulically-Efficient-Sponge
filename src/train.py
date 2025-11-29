"""
train.py
Trains the convolutional autoencoder on generated xylem-like structures.
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from model import XylemAutoencoder

# ---- CONFIG ----
DATA_DIR = "data/generated_microtubes"
RESULTS_DIR = "results/"
os.makedirs(RESULTS_DIR, exist_ok=True)
EPOCHS = 10
BATCH_SIZE = 4
LR = 1e-3
LATENT_DIM = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- DATASET ----
class XylemDataset(Dataset):
    def __init__(self, path):
        self.files = glob.glob(os.path.join(path, "*.png"))
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx])
        return self.transform(img)

# ---- LOAD DATA ----
dataset = XylemDataset(DATA_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---- MODEL ----
model = XylemAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ---- TRAIN LOOP ----
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        recon, _ = model(batch)
        loss = criterion(recon, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.5f}")

    # Save sample reconstruction
    with torch.no_grad():
        recon_img = recon[0].cpu().numpy().squeeze()
        orig_img = batch[0].cpu().numpy().squeeze()
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(orig_img, cmap="gray"); axs[0].set_title("Original")
        axs[1].imshow(recon_img, cmap="gray"); axs[1].set_title("Reconstructed")
        for ax in axs: ax.axis("off")
        plt.savefig(os.path.join(RESULTS_DIR, f"recon_epoch_{epoch+1}.png"))
        plt.close()

# ---- SAVE MODEL ----
torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "xylem_autoencoder.pt"))
print(f"✅ Training complete. Results saved to {RESULTS_DIR}")
