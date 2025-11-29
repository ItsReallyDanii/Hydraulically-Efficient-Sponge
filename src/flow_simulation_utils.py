import numpy as np
import torch
import torch.nn.functional as F

# ==================================================
# 💧 Compute Flow Physics Metrics
# ==================================================
def compute_flow_metrics(img, grad_scale=1.0):
    """
    Compute approximate flow physics metrics for a 2D xylem microstructure image.

    Args:
        img (np.ndarray): Grayscale normalized image (0–1).
        grad_scale (float): Optional multiplier to amplify flow sensitivity.

    Returns:
        dict: {
            "K": mean permeability proxy,
            "Porosity": fraction of open space,
            "Anisotropy": flow direction anisotropy
        }
    """

    # Ensure normalized float
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Convert to tensor
    tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # shape (1,1,H,W)

    # Gradient fields
    gy, gx = torch.gradient(tensor[0, 0])

    # Amplify flow gradient sensitivity
    grad_mag = torch.sqrt(gx ** 2 + gy ** 2) * grad_scale

    # Porosity = fraction of open (bright) pixels
    porosity = tensor.mean().item()

    # Effective permeability proxy (inverse relation with gradient magnitude)
    K = (1.0 / (1.0 + grad_mag * (1 - porosity))).mean().item()

    # Directional anisotropy ratio (|∂x| vs |∂y|)
    anisotropy = (gx.abs().mean() / (gy.abs().mean() + 1e-8)).item()

    return {
        "K": K,
        "Porosity": porosity,
        "Anisotropy": anisotropy,
    }
