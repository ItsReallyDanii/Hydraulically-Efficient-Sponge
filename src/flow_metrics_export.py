"""
flow_metrics_export.py
-------------------------------------
Runs Darcy flow simulation and exports quantitative metrics
for real and synthetic xylem structures.
"""

import os
import csv
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.sparse import diags, linalg

REAL_DIR = "data/real_xylem_preprocessed"
SYN_DIR = "data/generated_microtubes"
SAVE_DIR = "results/flow_metrics"
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE = (128, 128)
EPS = 1e-6


def load_image(path):
    img = Image.open(path).convert("L").resize(IMG_SIZE)
    return np.array(img, dtype=np.float32) / 255.0


def permeability_map(img):
    return gaussian_filter(img**3 + EPS, sigma=1)


def solve_darcy(k_map, delta_p=1.0, mu=1.0):
    ny, nx = k_map.shape
    N = nx * ny
    kx = (k_map[:, 1:] + k_map[:, :-1]) / 2
    ky = (k_map[1:, :] + k_map[:-1, :]) / 2

    main = np.zeros(N)
    east = np.zeros(N - 1)
    west = np.zeros(N - 1)
    north = np.zeros(N - nx)
    south = np.zeros(N - nx)

    for y in range(ny):
        for x in range(nx):
            i = y * nx + x
            k_e = kx[y, x] if x < nx - 1 else 0
            k_w = kx[y, x - 1] if x > 0 else 0
            k_n = ky[y - 1, x] if y > 0 else 0
            k_s = ky[y, x] if y < ny - 1 else 0
            main[i] = -(k_e + k_w + k_n + k_s)
            if x > 0:
                east[i - 1] = k_e
            if x < nx - 1:
                west[i] = k_w
            if y > 0:
                north[i - nx] = k_n
            if y < ny - 1:
                south[i] = k_s

    A = diags([main, east, west, north, south], [0, -1, 1, -nx, nx], format="csr")

    b = np.zeros(N)
    for x in range(nx):
        b[x] = delta_p * k_map[0, x]

    p = linalg.spsolve(A, b)
    p_field = p.reshape(ny, nx)

    grad_y, grad_x = np.gradient(p_field)
    vx = -k_map * grad_x / mu
    vy = -k_map * grad_y / mu
    return p_field, vx, vy


def porosity(img):
    return np.mean(img)


def anisotropy(img):
    gx, gy = np.gradient(img)
    return np.mean(np.abs(gx)) / (np.mean(np.abs(gy)) + EPS)


def process_folder(folder, label, writer):
    print(f"\nProcessing {label}...")
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".png")])
    for f in files:
        path = os.path.join(folder, f)
        img = load_image(path)
        k_map = permeability_map(img)
        p, vx, vy = solve_darcy(k_map)

        flow_rate = np.mean(np.abs(vy))
        mean_k = np.mean(k_map)
        mean_grad = np.mean(np.abs(np.gradient(p)[0]))
        phi = porosity(img)
        aniso = anisotropy(img)

        writer.writerow([label, f, mean_k, mean_grad, flow_rate, phi, aniso])


if __name__ == "__main__":
    csv_path = os.path.join(SAVE_DIR, "flow_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Type", "Filename", "Mean_K", "Mean_dP/dy", "FlowRate", "Porosity", "Anisotropy"])

        process_folder(REAL_DIR, "Real", writer)
        process_folder(SYN_DIR, "Synthetic", writer)

    print(f"\n✅ Exported all metrics to {csv_path}")
