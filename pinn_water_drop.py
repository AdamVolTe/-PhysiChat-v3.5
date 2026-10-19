"""Object sinking in water with buoyancy and viscous drag:  m y'' = m g - F_b - b y'"""
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

m = 2.0               # mass (kg)
V = 0.001             # volume (m^3). NOTE: with V=0.002 buoyancy equals weight and nothing moves
rho_water = 1000.0
g = 9.81
F_b = rho_water * V * g     # Archimedes buoyancy force = 9.81 N
b_water = 0.5               # viscous drag coefficient
T_END = 3.0
SCALE = 20.0                # typical depth scale (m)


class PINNWaterDrop(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, t):
        s = t / T_END
        # Hard constraint: y(0)=0 and y'(0)=0 hold exactly
        return s**2 * SCALE * self.net(s)


def exact_solution(t):
    a = (m * g - F_b) / m
    tau = m / b_water
    return a * tau**2 * (np.exp(-t / tau) - 1) + a * tau * t


def train(epochs=8000, lr=2e-3, verbose=True):
    torch.manual_seed(0)
    model = PINNWaterDrop().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    t = torch.linspace(0, T_END, 200).view(-1, 1).to(device).requires_grad_(True)

    for step in range(epochs + 1):
        opt.zero_grad()
        y = model(t)
        dy = torch.autograd.grad(y, t, torch.ones_like(y), create_graph=True)[0]
        d2y = torch.autograd.grad(dy, t, torch.ones_like(dy), create_graph=True)[0]
        res = (m * d2y - (m * g - F_b) + b_water * dy) / g
        loss = torch.mean(res**2)
        loss.backward()
        opt.step()
        sched.step()
        if verbose and step % max(1, epochs // 5) == 0:
            print(f"[water] step {step:6d}/{epochs} | loss {loss.item():.3e}")

    t_test = np.linspace(0, T_END, 200)
    with torch.no_grad():
        y_pred = model(torch.tensor(t_test, dtype=torch.float32).view(-1, 1).to(device)).cpu().numpy().flatten()
    err = float(np.mean(np.abs(y_pred - exact_solution(t_test))))
    if verbose:
        print(f"[water] mean depth error: {err:.4f} m")
    return model, t_test, y_pred, err


def plot(t, y_pred, show=True, path="water_drop.png"):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(t, exact_solution(t), "k-", lw=2, label="Exact solution")
    plt.plot(t, y_pred, "b--", lw=2, label="PINN prediction")
    plt.title("Object sinking in water (buoyancy + viscous drag)")
    plt.xlabel("Time t (s)")
    plt.ylabel("Depth y (m)")
    plt.legend()
    plt.grid(True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()


if __name__ == "__main__":
    _, t, y, _ = train()
    plot(t, y)
