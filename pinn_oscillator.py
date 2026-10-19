"""1D damped harmonic oscillator:  u'' + 2*d*u' + w0^2 * u = 0,  u(0)=1, u'(0)=0"""
import numpy as np
import torch
import torch.nn as nn

D, W0 = 2.0, 20.0   # damping and natural frequency
T_END = 1.0


def exact_solution(t, d=D, w0=W0):
    w = np.sqrt(w0**2 - d**2)
    phi = np.arctan(-d / w)
    A = 1 / (2 * np.cos(phi))
    return 2 * A * np.exp(-d * t) * np.cos(w * t + phi)


class FCN(nn.Module):
    def __init__(self, hidden=64, layers=4):
        super().__init__()
        blocks, dim = [], 1
        for _ in range(layers):
            blocks += [nn.Linear(dim, hidden), nn.Tanh()]
            dim = hidden
        blocks.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, t):
        # Hard constraint: u(0)=1 and u'(0)=0 hold exactly for any weights
        return 1.0 + (t / T_END) ** 2 * self.net(t / T_END)


def train(steps=20000, lr=2e-3, seed=42, verbose=True):
    torch.manual_seed(seed)
    model = FCN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)
    t_phys = torch.linspace(0, T_END, 200).view(-1, 1).requires_grad_(True)

    for i in range(steps + 1):
        opt.zero_grad()
        u = model(t_phys)
        du = torch.autograd.grad(u, t_phys, torch.ones_like(u), create_graph=True)[0]
        d2u = torch.autograd.grad(du, t_phys, torch.ones_like(du), create_graph=True)[0]
        residual = (d2u + 2 * D * du + W0**2 * u) / W0**2   # normalized
        loss = torch.mean(residual**2)
        loss.backward()
        opt.step()
        sched.step()
        if verbose and i % max(1, steps // 5) == 0:
            print(f"[oscillator] step {i:6d} | physics loss {loss.item():.3e}")

    t_test = np.linspace(0, T_END, 200)
    with torch.no_grad():
        u_pinn = model(torch.tensor(t_test, dtype=torch.float32).view(-1, 1)).numpy().flatten()
    err = float(np.mean(np.abs(u_pinn - exact_solution(t_test))))
    if verbose:
        print(f"[oscillator] mean abs error vs exact solution: {err:.4f}")
    return model, t_test, u_pinn, err


def plot(t, u_pinn, show=True, path="oscillator.png"):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(t, exact_solution(t), "k-", alpha=0.6, lw=2, label="Exact solution")
    plt.plot(t, u_pinn, "r--", lw=2, label="PINN prediction")
    plt.title("PINN - Damped Harmonic Oscillator")
    plt.xlabel("Time (s)")
    plt.ylabel("Position u(t)")
    plt.legend()
    plt.grid(True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()


if __name__ == "__main__":
    _, t, u, _ = train()
    plot(t, u)
