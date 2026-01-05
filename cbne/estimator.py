# cbne/estimator.py
import torch
from .clique_utils import build_clique_structures

def run_batch_onfly_gpu(diag, off, idx, sgn, cnt, n_vertices, k, z, B, device):
    F = diag.shape[0]
    start = torch.randint(0, F, (B,), device=device)
    cur = start.clone()
    prod = torch.ones(B, device=device)

    for _ in range(z):
        diag_j = diag[cur]
        c_j = cnt[cur]
        off_j = c_j.float() / n_vertices
        colnorm = diag_j + off_j
        safe = torch.where(colnorm == 0, torch.ones_like(colnorm), colnorm)
        jump_prob = torch.where(c_j > 0, 1 - diag_j / safe, torch.zeros_like(colnorm))
        prod *= colnorm
        u = torch.rand(B, device=device)
        jump = (u < jump_prob) & (c_j > 0)

        if jump.any():
            jf = cur[jump]
            cc = cnt[jf]
            base = off[jf]
            r = (torch.rand(len(cc), device=device) * cc.float()).long()
            flat = base + r
            cur_new = idx[flat]
            s_new = sgn[flat].float()
            cur[jump] = cur_new
            prod[jump] *= s_new

    return torch.where(cur == start, prod, torch.zeros_like(prod))

def estimate_onfly_gpu(G, k, z, num_samples, batch=4096, device="cuda"):
    faces, diag_np, off_np, idx_np, sgn_np, cnt_np = build_clique_structures(G, k)
    if len(faces) == 0: return 0.0

    diag = torch.tensor(diag_np, device=device)
    off = torch.tensor(off_np, device=device)
    idx = torch.tensor(idx_np, device=device)
    sgn = torch.tensor(sgn_np, device=device)
    cnt = torch.tensor(cnt_np, device=device)

    acc = []
    for _ in range((num_samples + batch - 1) // batch):
        cur_batch = min(batch, num_samples)
        samples = run_batch_onfly_gpu(diag, off, idx, sgn, cnt, G.n, k, z, cur_batch, device)
        acc.append(samples.mean().item())
        num_samples -= cur_batch

    return float(np.mean(acc))

