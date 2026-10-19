"""2D projectile with linear air drag:  m x'' + b x' = 0,  m y'' + b y' + m g = 0"""
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

g, b, m = 9.81, 0.15, 1.0
v0, angle = 50.0, 45.0
theta = np.radians(angle)
vx0, vy0 = v0 * np.cos(theta), v0 * np.sin(theta)
T_END = 5.0
SCALE = 100.0   # typical length scale (m), keeps network outputs near 1


class PINNProjectile(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2),
        )

    def forward(self, t):
        s = t / T_END
        out = self.net(s)
        # Hard constraint: x(0)=y(0)=0 and x'(0)=vx0, y'(0)=vy0 hold exactly
        x = vx0 * t + s**2 * SCALE * out[:, 0:1]
        y = vy0 * t + s**2 * SCALE * out[:, 1:2]
        return x, y


def exact_trajectory(t):
    x = (m * vx0 / b) * (1 - np.exp(-b * t / m))
    y = (m / b) * (vy0 + (m * g / b)) * (1 - np.exp(-b * t / m)) - (m * g * t / b)
    return x, y


def train(epochs=10000, lr=2e-3, verbose=True):
    torch.manual_seed(0)
    model = PINNProjectile().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    t = torch.linspace(0, T_END, 200).view(-1, 1).to(device).requires_grad_(True)

    for step in range(epochs + 1):
        opt.zero_grad()
        x, y = model(t)
        dx = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
        dy = torch.autograd.grad(y, t, torch.ones_like(y), create_graph=True)[0]
        d2x = torch.autograd.grad(dx, t, torch.ones_like(dx), create_graph=True)[0]
        d2y = torch.autograd.grad(dy, t, torch.ones_like(dy), create_graph=True)[0]
        res_x = (m * d2x + b * dx) / g
        res_y = (m * d2y + b * dy + m * g) / g
        loss = torch.mean(res_x**2) + torch.mean(res_y**2)
        loss.backward()
        opt.step()
        sched.step()
        if verbose and step % max(1, epochs // 5) == 0:
            print(f"[projectile] step {step:6d}/{epochs} | loss {loss.item():.3e}")

    t_test = np.linspace(0, T_END, 200)
    with torch.no_grad():
        xp, yp = model(torch.tensor(t_test, dtype=torch.float32).view(-1, 1).to(device))
    x_pred, y_pred = xp.cpu().numpy().flatten(), yp.cpu().numpy().flatten()
    x_ex, y_ex = exact_trajectory(t_test)
    err = float(np.mean(np.hypot(x_pred - x_ex, y_pred - y_ex)))
    if verbose:
        print(f"[projectile] mean position error: {err:.3f} m")
    return model, (x_pred, y_pred), (x_ex, y_ex), err


def plot(pred, exact, show=True, path="projectile.png"):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(*exact, "k-", lw=2, label="Exact trajectory")
    plt.plot(*pred, "r--", lw=2, label="PINN prediction")
    plt.title("Projectile trajectory with air drag")
    plt.xlabel("Horizontal distance X (m)")
    plt.ylabel("Vertical distance Y (m)")
    plt.legend()
    plt.grid(True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()


if __name__ == "__main__":
    _, pred, exact, _ = train()
    plot(pred, exact)
