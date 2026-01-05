# cbne/moment_features.py
from .estimator import estimate_onfly_gpu
import numpy as np

def compute_moment_feature_map_gpu_filtration(G_nx, f_values, num_alphas=10, z_max=5, samples=30000, k_clique=1, device="cuda"):
    phi = []
    for _ in range(num_alphas):
        for z in range(1, z_max + 1):
            est = estimate_onfly_gpu(G_nx, k_clique, z, samples, device=device)
            phi.append(est)
    return np.array(phi, dtype=np.float32)

