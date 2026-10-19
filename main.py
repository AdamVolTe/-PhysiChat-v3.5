"""Runs all standalone PINN demos and saves a plot for each (no windows opened)."""
import pinn_oscillator
import pinn_projectile
import pinn_water_drop

if __name__ == "__main__":
    print("=== PhysicsPINN: running all engines ===")

    print("\n1. 1D Damped Harmonic Oscillator")
    _, t, u, err1 = pinn_oscillator.train()
    pinn_oscillator.plot(t, u, show=False)

    print("\n2. 2D Projectile Motion with Drag")
    _, pred, exact, err2 = pinn_projectile.train()
    pinn_projectile.plot(pred, exact, show=False)

    print("\n3. Water Drop (buoyancy + drag)")
    _, t, y, err3 = pinn_water_drop.train()
    pinn_water_drop.plot(t, y, show=False)

    print("\n=== Summary ===")
    print(f"Oscillator mean error : {err1:.4f}")
    print(f"Projectile mean error : {err2:.3f} m")
    print(f"Water drop mean error : {err3:.4f} m")
    print("Plots saved: oscillator.png, projectile.png, water_drop.png")
