"""
map_latent_to_physics.py
Visualize how learned latent representations correlate with physical performance metrics.
"""

import os, sys, subprocess, numpy as np, torch, matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# dependency check
REQUIRED = ["torch", "matplotlib", "numpy", "scikit-learn", "Pillow"]
for pkg in REQUIRED:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results", "latent_physics_map")
os.makedirs(RESULTS_DIR, exist_ok=True)

# import model + physics
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from src.model import XylemAutoencoder
from src.simulate_flow import simulate_pressure_field, compute_conductivity

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_DIM = 32
SAMPLES = 50

# --- main ---
def main():
    model = XylemAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
    dummy = torch.zeros(1,1,256,256).to(DEVICE); _ = model(dummy)
    state_dict = torch.load(os.path.join(ROOT_DIR, "results", "xylem_autoencoder.pt"), map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    zs, conductivities = [], []

    for _ in range(SAMPLES):
        z = torch.randn(1, LATENT_DIM, device=DEVICE)
        recon = model.decoder_deconv(model.fc_dec(z).view(-1,128,16,16))
        img = recon.detach().cpu().numpy()[0,0]
        p_field, mask = simulate_pressure_field(img)
        cond = compute_conductivity(p_field, mask)
        zs.append(z.cpu().numpy().flatten())
        conductivities.append(cond)

    Z = np.array(zs)
    conductivities = np.array(conductivities)
    Z_scaled = StandardScaler().fit_transform(Z)

    # dimensionality reduction
    emb = TSNE(n_components=2, perplexity=15, learning_rate="auto", init="random").fit_transform(Z_scaled)

    # visualization
    plt.figure(figsize=(6,5))
    sc = plt.scatter(emb[:,0], emb[:,1], c=conductivities, cmap="viridis", s=60, alpha=0.8)
    plt.colorbar(sc, label="Conductivity")
    plt.title("Latent Space vs Physical Conductivity")
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "latent_physics_map.png"), dpi=200)
    plt.close()

    print("✅ Latent-Physics mapping complete.")
    print(f"Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
