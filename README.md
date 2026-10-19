# PhysiChat ⚛️

Educational PyTorch project on **Physics-Informed Neural Networks (PINNs)**, with a chat interface
(**Physics AI Copilot**): describe a system in Darija, Arabic, French or English and the app runs a live
simulation, analyses it and explains the result.

Developed by **Adam** (15-year-old AI & Hard Tech developer from Morocco).

> This is a learning/experimental project. For simple ODEs like these, classical solvers (e.g. SciPy)
> are faster and more accurate than a PINN, so **every plot compares the PINN against a numerical reference**.
> PINNs become interesting for inverse problems, noisy or missing data and instant parametric surrogates.

## Features

- 💬 **Copilot Chat**: natural-language requests, Claude tool calling, answers in your language
- 🎛️ **Playground**: sliders for damping and time horizon, instant results
- ⚖️ **Compare**: overlay several damping values (PINN vs SciPy)
- 🧪 **Model Lab**: error heatmap over (alpha, T), real training loss, train/retrain from the UI
- 📊 Per-simulation analysis: damping regime, settling time, oscillation period, phase portrait, energy, error
- ⬇ Export results as CSV or a real PDF report
- 4 physics systems: damped oscillator, non-linear pendulum, thermal cooling, driven (forced) oscillator

## Project structure

| File | What it does |
|------|--------------|
| `app.py` | Streamlit app (chat, playground, compare, model lab) |
| `chat_agent.py` | Claude tool-calling agent (+ offline parser fallback) |
| `physics_engine.py` | Parametric PINN (inputs: t and alpha) with exact initial conditions |
| `physics_trainer.py` | Trains one model per system, saves the real loss history |
| `analysis.py` | Regime, settling time, period, energy (NumPy only) |
| `pinn_oscillator.py`, `pinn_projectile.py`, `pinn_water_drop.py` | Standalone PINN demos |
| `main.py` | Runs the three standalone demos |

## Quickstart

```bash
git clone https://github.com/AdamMeveloper/PhysicsPINN.git
cd PhysicsPINN
pip install -r requirements.txt

python physics_trainer.py     # train the models used by the app (a few minutes on CPU)
streamlit run app.py          # open the app
python main.py                # optional: standalone demos
```

To get natural-language answers from Claude, set your API key before launching:

```bash
export GROQ_API_KEY="your-key"     # Windows PowerShell: $env:ANTHROPIC_API_KEY="your-key"
```

Without a key the app still works with a simple keyword parser. Never commit your key to Git.

## How the chat works

1. You write: "شنو يوقع لنواس تخميدو 0.8 فـ 15 ثانية؟"
2. Claude calls the `run_simulation(system, alpha, horizon)` tool.
3. The app runs the trained PINN and compares it with a SciPy reference.
4. Claude explains the result using the real numbers returned.
