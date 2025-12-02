import os
from PIL import Image
import numpy as np

DATA_DIR = "data/real_xylem_preprocessed"

files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".png")]
files = sorted(files)[:10]

print(f"Checking {len(files)} images from {DATA_DIR}...")

means = []
for fname in files:
    path = os.path.join(DATA_DIR, fname)
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0  # 0–1
    m = float(arr.mean())
    means.append(m)
    print(f"{fname}: mean pixel = {m:.4f}")

if means:
    print(f"\nAverage of means: {np.mean(means):.4f}")
import pandas as pd

csv_path = "results/flow_metrics/flow_metrics.csv"
df = pd.read_csv(csv_path)

# Adjust these if your column names differ
name_col = [c for c in df.columns if "file" in c.lower()][0]
poro_col = [c for c in df.columns if "porosity" in c.lower()][0]

mask = df[name_col].isin(
    ["F02a.png","F02b.png","F02c.png","F02d.png","F02e.png",
     "F03a.png","F03b.png","F03c.png","F03d.png","F03e.png"]
)

print(df.loc[mask, [name_col, poro_col]])
print("\nMean solver porosity on those:", df.loc[mask, poro_col].mean())
