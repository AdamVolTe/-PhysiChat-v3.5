PhysiChat ⚛️
Educational PyTorch project on Physics-Informed Neural Networks (PINNs), with a chat interface
(Physics AI Copilot): describe a system in Darija, Arabic, French or English and the app runs a live
simulation, analyses it and explains the result.
Developed by Adam (15-year-old AI & Hard Tech developer from Morocco).
This is a learning/experimental project. For simple ODEs like these, classical solvers (e.g. SciPy)
are faster and more accurate than a PINN, so every plot compares the PINN against a numerical reference.
PINNs become interesting for inverse problems, noisy or missing data and instant parametric surrogates.
Features
💬 Copilot Chat: natural-language requests, LLM tool calling (Groq), answers in your language
🎛️ Playground: sliders for damping and time horizon, instant results
⚖️ Compare: overlay several damping values (PINN vs SciPy)
🧪 Model Lab: error heatmap over (alpha, T), real training loss, train/retrain from the UI
ℹ️ About & Feedback: feedback form, optional waitlist, privacy notice
📊 Per-simulation analysis: damping regime, settling time, oscillation period, phase portrait, energy, error
⬇ Export results as CSV or a real PDF report
4 physics systems: damped oscillator, non-linear pendulum, thermal cooling, driven (forced) oscillator
Project structure
File
What it does
app.py
Streamlit app (chat, playground, compare, model lab)
chat_agent.py
Groq tool-calling agent (+ offline parser fallback)
physics_engine.py
Parametric PINN (inputs: t and alpha) with exact initial conditions
physics_trainer.py
Trains one model per system, saves the real loss history
analysis.py
Regime, settling time, period, energy (NumPy only)
pinn_oscillator.py, pinn_projectile.py, pinn_water_drop.py
Standalone PINN demos
main.py
Runs the three standalone demos

git clone https://github.com/AdamMeveloper/PhysicsPINN.git
cd PhysicsPINN
pip install -r requirements.txt

python physics_trainer.py     # train the models used by the app (a few minutes on CPU)
streamlit run app.py          # open the app
python main.py                # optional: standalone demos

GROQ_API_KEY=your-key            # chat + tool calling (https://console.groq.com)
GROQ_MODEL=openai/gpt-oss-120b   # optional: see console.groq.com/docs/models for model IDs
SUPABASE_URL=...                 # optional: fair-use limits, feedback, waitlist (run supabase_schema.sql once)
SUPABASE_KEY=...
FAIR_USE_LIMIT=10                # optional: messages per 24h per visitor (protects the free Groq quota)
GROQ_FALLBACK_MODEL=openai/gpt-oss-20b   # optional: used when the main model hits 429
CONTACT_EMAIL=you@example.com    # optional: shown on the About page
ALLOW_TRAINING=1                 # optional: enables the Train button (local use only)
(see .env.example)

Without a Groq key the app still works: a keyword parser runs the simulations (offline mode).
The app is a free beta. Without Supabase the fair-use limit is per browser session and feedback is saved to data/*.jsonl.
Note: models/*.pth must be present on the server (commit them or train first), otherwise simulations fail.
On small hosts install the CPU build of torch: pip install torch --index-url https://download.pytorch.org/whl/cpu
How the chat works
You write: "شنو يوقع لنواس تخميدو 0.8 فـ 15 ثانية؟"
The Groq LLM calls the run_simulation(system, alpha, horizon) tool.
The app runs the trained PINN and compares it with a SciPy reference.
The LLM explains the result using the real numbers returned.