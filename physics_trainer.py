"""
Trains one parametric PINN per physics domain.
Saves weights to models/<domain>.pth and the real loss history to models/<domain>_loss.npy
"""
import os
import sys
import time

import numpy as np
import torch

from physics_engine import (ALPHA_MAX, DOMAINS, T_MAX, PhysicsEngineCore,
                            device, reference_solution)

MODELS_DIR = "models"


def train_domain(domain, epochs=6000, batch=1024, lr=2e-3, verbose=True):
    torch.manual_seed(0)
    model = PhysicsEngineCore(domain)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    history = []
    start = time.time()

    for epoch in range(epochs):
        # random collocation points over the whole (t, alpha) domain
        t = torch.rand(batch, 1, device=device) * T_MAX
        alpha = torch.rand(batch, 1, device=device) * ALPHA_MAX

        opt.zero_grad()
        _, res = model.residual(t, alpha)
        loss = torch.mean(res**2)
        loss.backward()
        opt.step()
        sched.step()
        history.append(loss.item())

        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"[{domain}] epoch {epoch+1}/{epochs}  loss={loss.item():.3e}"
                  f"  ({time.time()-start:.0f}s)")

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{domain}.pth"))
    np.save(os.path.join(MODELS_DIR, f"{domain}_loss.npy"), np.array(history))
    return model, history


def evaluate(model, domain, alphas=(0.2, 0.8, 1.5), horizon=10.0):
    t = np.linspace(0, horizon, 300)
    out = {}
    for a in alphas:
        err = np.abs(model.predict(t, a) - reference_solution(domain, t, a))
        out[a] = (err.mean(), err.max())
    return out


if __name__ == "__main__":
    targets = sys.argv[1:] or list(DOMAINS)
    for d in targets:
        print(f"\n=== Training {d} ===")
        m, _ = train_domain(d)
        for a, (mean_e, max_e) in evaluate(m, d).items():
            print(f"  alpha={a}: mean abs error={mean_e:.4f}, max={max_e:.4f}")
    print("\nDone. Models saved in ./models")
