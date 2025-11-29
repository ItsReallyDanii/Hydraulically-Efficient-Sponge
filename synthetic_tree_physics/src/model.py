"""
model.py
Defines and trains a simple convolutional autoencoder for xylem-like microstructures.
"""

import torch
import torch.nn as nn

class XylemAutoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        # Encoder: compresses 256x256 image → latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),   # 256 → 128
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 128 → 64
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 64 → 32
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, latent_dim)
        )

        # Decoder: reconstructs image from latent vector
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 32 * 32),
            nn.ReLU(),
            nn.Unflatten(1, (64, 32, 32)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # 32 → 64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # 64 → 128
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # 128 → 256
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z
