"""
Pure NumPy/SciPy helpers that turn a trajectory y(t) into engineering metrics.
No PyTorch needed, so this module is easy to test on its own.
"""
import numpy as np


def velocity(t, y):
    return np.gradient(y, t)


def energy(domain, t, y):
    """Mechanical energy per unit mass (oscillator-like systems)."""
    v = velocity(t, y)
    potential = (1.0 - np.cos(y)) if domain == "pendulum" else 0.5 * y**2
    return 0.5 * v**2 + potential


def regime(domain, alpha):
    """Returns (name, short explanation) for the damping regime."""
    if domain == "thermal":
        if alpha <= 1e-9:
            return "No cooling", "alpha = 0: temperature stays constant"
        return "First-order decay", f"time constant = 1/alpha = {1.0 / alpha:.2f} s"
    suffix = " (non-linear)" if domain == "pendulum" else ""
    if domain == "driven":
        base = "Forced response"
        note = "transient + steady oscillation at the driving frequency"
        return base, note
    if alpha <= 1e-9:
        return "Undamped" + suffix, "energy is conserved, oscillates forever"
    if alpha < 1.95:
        return "Underdamped" + suffix, "oscillates while the amplitude decays"
    if alpha <= 2.05:
        return "Critically damped" + suffix, "fastest return without oscillating"
    return "Overdamped" + suffix, "slow return, no oscillation"


def settling_time(t, y, tol=0.02):
    """First time after which |y| stays within tol * max|y|. None if it never settles."""
    peak = float(np.max(np.abs(y)))
    if peak < 1e-12:
        return 0.0
    outside = np.where(np.abs(y) > tol * peak)[0]
    if len(outside) == 0:
        return 0.0
    last = outside[-1]
    if last >= len(t) - 1:
        return None
    return float(t[last + 1])


def estimate_period(t, y):
    """
    Period from zero crossings (two crossings per period), with linear interpolation.
    Needs at least 3 crossings for a stable estimate; returns None otherwise
    (e.g. overdamped motion, or a horizon too short to see enough oscillations).
    """
    s = np.sign(y)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if len(idx) < 3:
        return None
    t0, t1, y0, y1 = t[idx], t[idx + 1], y[idx], y[idx + 1]
    crossings = t0 - y0 * (t1 - t0) / (y1 - y0)
    return float(2.0 * np.mean(np.diff(crossings)))


def summarize(domain, alpha, t, y):
    """JSON-friendly summary (plain Python types only)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    name, note = regime(domain, alpha)
    st_time = None if domain == "driven" else settling_time(t, y)
    return {
        "regime": name,
        "regime_note": note,
        "peak_abs": float(np.max(np.abs(y))),
        "final_value": float(y[-1]),
        "settling_time_s": st_time,
        "period_s": estimate_period(t, y) if domain != "thermal" else None,
    }
