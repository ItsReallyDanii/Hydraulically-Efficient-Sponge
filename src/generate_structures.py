import os
import argparse
import torch
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm
from src.model import XylemAutoencoder  # or your model class
import pandas as pd

def generate_structures(model_path, n=64, out_dir="data/generated_microtubes"):
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    model = XylemAutoencoder()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # Create latent samples
    latent_dim = model.latent_dim if hasattr(model, "latent_dim") else 128
    z = torch.randn(n, latent_dim)

    generated = []
    print(f"🧩 Generating {n} synthetic structures...")
    for i in tqdm(range(n)):
        with torch.no_grad():
            img = model.decode(z[i].unsqueeze(0)).cpu()
        img = img.squeeze().numpy()

        # Normalize & enhance contrast
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = np.clip(img * 1.2, 0, 1)  # slight contrast boost
        img_torch = torch.tensor(img).unsqueeze(0)

        # Save
        out_path = os.path.join(out_dir, f"synthetic_{i+1:03d}.png")
        save_image(img_torch, out_path)

        # Log metrics
        porosity = float((img > 0.5).mean())
        intensity_mean = float(img.mean())
        intensity_var = float(img.var())

        generated.append({
            "filename": f"synthetic_{i+1:03d}.png",
            "porosity_est": porosity,
            "intensity_mean": intensity_mean,
            "intensity_var": intensity_var
        })

    # Save metadata
    log_path = os.path.join(out_dir, "generation_log.csv")
    pd.DataFrame(generated).to_csv(log_path, index=False)
    print(f"✅ Generated {n} structures in {out_dir}")
    print(f"🧾 Generation log saved → {log_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to tuned model (.pth)")
    parser.add_argument("--n", type=int, default=64, help="Number of structures to generate")
    args = parser.parse_args()

    generate_structures(args.model, args.n)

if __name__ == "__main__":
    main()
