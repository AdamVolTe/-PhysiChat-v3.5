"""
Physics engine: a parametric PINN (Physics-Informed Neural Network).

The network takes (t, alpha) as input, so ONE trained model covers a whole
range of damping values. Initial conditions are enforced exactly through a
"hard constraint" in the output (no way for the net to collapse to y = 0).

Supported domains:
  oscillator : y'' + alpha*y' + y = 0           y(0)=1, y'(0)=0
  pendulum   : y'' + alpha*y' + sin(y) = 0      y(0)=1.5 rad, y'(0)=0
  thermal    : y' + alpha*y = 0 (lumped cooling) y(0)=1
  driven     : y'' + alpha*y' + y = cos(1.2 t)   y(0)=0, y'(0)=0
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T_MAX = 20.0      # maximum time horizon the model is trained on
ALPHA_MAX = 2.0   # maximum damping coefficient the model is trained on

DOMAINS = {
    "oscillator": {
        "label": "Damped Harmonic Oscillator",
        "order": 2, "y0": 1.0, "v0": 0.0,
    },
    "pendulum": {
        "label": "Non-linear Pendulum System",
        "order": 2, "y0": 1.5, "v0": 0.0,
    },
    "thermal": {
        "label": "Thermal Cooling (lumped)",
        "order": 1, "y0": 1.0, "v0": 0.0,
    },
    "driven": {
        "label": "Driven (Forced) Oscillator",
        "order": 2, "y0": 0.0, "v0": 0.0,
        "force": (1.0, 1.2),   # amplitude F and driving frequency w
    },
}


class PhysicsEngineCore(nn.Module):
    def __init__(self, domain="oscillator", hidden=64, depth=3):
        super().__init__()
        if domain not in DOMAINS:
            raise ValueError(f"Unknown domain: {domain}")
        self.domain = domain
        cfg = DOMAINS[domain]
        self.order, self.y0, self.v0 = cfg["order"], cfg["y0"], cfg["v0"]

        layers, dim_in = [], 2
        for _ in range(depth):
            layers += [nn.Linear(dim_in, hidden), nn.Tanh()]
            dim_in = hidden
        layers.append(nn.Linear(dim_in, 1))
        self.network = nn.Sequential(*layers)
        self.to(device)

    def forward(self, t, alpha):
        """t, alpha: tensors of shape (N, 1). Initial conditions hold exactly."""
        s = t / T_MAX
        net = self.network(torch.cat([s, alpha / ALPHA_MAX], dim=1))
        if self.order == 2:
            return self.y0 + self.v0 * t + s**2 * net
        return self.y0 + s * net

    def residual(self, t, alpha):
        """Returns (y, physics residual) for the domain's ODE."""
        t = t.clone().detach().requires_grad_(True)
        y = self.forward(t, alpha)
        dy = torch.autograd.grad(y, t, torch.ones_like(y), create_graph=True)[0]
        if self.order == 1:
            return y, dy + alpha * y
        d2y = torch.autograd.grad(dy, t, torch.ones_like(dy), create_graph=True)[0]
        restoring = torch.sin(y) if self.domain == "pendulum" else y
        force = 0.0
        if self.domain == "driven":
            amp, freq = DOMAINS["driven"]["force"]
            force = amp * torch.cos(freq * t)
        return y, d2y + alpha * dy + restoring - force

    def solve_ode(self, t_tensor, alpha=1.0):
        """Predict y(t) for a scalar alpha. Returns (y, mean squared residual)."""
        t_tensor = t_tensor.to(device)
        a = torch.full_like(t_tensor, float(alpha))
        y, res = self.residual(t_tensor, a)
        return y, torch.mean(res**2)

    @torch.no_grad()
    def predict(self, t_np, alpha):
        t = torch.tensor(t_np, dtype=torch.float32, device=device).reshape(-1, 1)
        a = torch.full_like(t, float(alpha))
        return self.forward(t, a).cpu().numpy().flatten()


def reference_solution(domain, t_np, alpha):
    """Accurate numerical reference (scipy) used as ground truth."""
    cfg = DOMAINS[domain]
    t_np = np.asarray(t_np, dtype=float)

    def rhs(t, s):
        if cfg["order"] == 1:
            return [-alpha * s[0]]
        restoring = np.sin(s[0]) if domain == "pendulum" else s[0]
        force = 0.0
        if domain == "driven":
            amp, freq = cfg["force"]
            force = amp * np.cos(freq * t)
        return [s[1], -alpha * s[1] - restoring + force]

    s0 = [cfg["y0"]] if cfg["order"] == 1 else [cfg["y0"], cfg["v0"]]
    sol = solve_ivp(rhs, (0, t_np.max()), s0, t_eval=t_np, rtol=1e-9, atol=1e-11)
    return sol.y[0]


def analytic_oscillator(t, alpha):
    """Closed-form solution of y'' + alpha*y' + y = 0, y(0)=1, y'(0)=0."""
    t = np.asarray(t, dtype=float)
    disc = 1.0 - alpha**2 / 4.0
    if abs(disc) < 1e-9:  # critically damped
        return (1 + 0.5 * alpha * t) * np.exp(-0.5 * alpha * t)
    if disc > 0:  # underdamped
        w = np.sqrt(disc)
        return np.exp(-alpha * t / 2) * (np.cos(w * t) + alpha / (2 * w) * np.sin(w * t))
    r = np.sqrt(-disc)  # overdamped
    l1, l2 = -alpha / 2 + r, -alpha / 2 - r
    c1 = -l2 / (l1 - l2)
    c2 = l1 / (l1 - l2)
    return c1 * np.exp(l1 * t) + c2 * np.exp(l2 * t)


if __name__ == "__main__":
    engine = PhysicsEngineCore("oscillator")
    t = torch.linspace(0, 5, 5).reshape(-1, 1)
    y, loss = engine.solve_ode(t, alpha=0.5)
    print("Engine OK. y(0) =", float(y[0]), "(must be exactly 1.0)")
