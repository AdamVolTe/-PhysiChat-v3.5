import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from matplotlib.backends.backend_pdf import PdfPages

from analysis import energy, summarize, velocity
from chat_agent import chat_turn, get_api_key
from physics_engine import ALPHA_MAX, DOMAINS, PhysicsEngineCore, reference_solution
from physics_trainer import MODELS_DIR, train_domain
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

@st.cache_resource
def init_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

st.set_page_config(page_title="PhysiChat · General AI Copilot", page_icon="⚛️",
                   layout="wide", initial_sidebar_state="expanded")

# =====================================================================
# Settings & Dynamic Theme/Language Control (إعدادات اللغة والمظهر)
# =====================================================================
with st.sidebar:
    st.markdown("### ⚙️ الإعدادات / Settings")
    lang = st.selectbox("🌐 اللغة / Langue", ["الدارجة / العربية", "Français", "English"], key="user_lang")
    theme_mode = st.radio("🎨 المظهر / Thème", ["Dark Mode 🌙", "Light Mode ☀️"], key="user_theme")
    st.divider()

is_rtl = (lang == "الدارجة / العربية")
is_light = (theme_mode == "Light Mode ☀️")

# الألوان حسب المظهر (Dark vs Light)
bg_color = "#ffffff" if is_light else "radial-gradient(1200px 600px at 50% -10%, #17223a 0%, #090d16 62%)"
text_color = "#0f172a" if is_light else "#f1f5f9"
chat_bg = "#f1f5f9" if is_light else "rgba(30,41,59,.30)"
chat_border = "#cbd5e1" if is_light else "rgba(255,255,255,.06)"
text_dir = "rtl" if is_rtl else "ltr"
text_align = "right" if is_rtl else "left"

COL = {"pinn": "#38bdf8", "ref": "#34d399", "amber": "#f59e0b", "violet": "#a78bfa", "red": "#f87171"}
PALETTE = ["#38bdf8", "#a78bfa", "#f59e0b", "#34d399", "#f472b6", "#fb923c", "#22d3ee"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: {bg_color} !important; color: {text_color} !important; }}
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.main .block-container {{ padding-top: 1.6rem; max-width: 1250px; direction: {text_dir}; text-align: {text_align}; }}

section[data-testid="stSidebar"] {{ background: rgba(10,14,22,0.97) !important;
    border-right: 1px solid rgba(255,255,255,0.07); color: #f1f5f9 !important; }}
.brand {{ display:flex; align-items:center; gap:10px; margin-bottom:2px; }}
.brand-logo {{ width:34px; height:34px; border-radius:10px; display:flex; align-items:center; justify-content:center;
    background: linear-gradient(135deg,#0ea5e9,#6366f1); font-size:18px; box-shadow:0 4px 16px rgba(14,165,233,.35); }}
.brand-name {{ font-weight:800; font-size:18px; letter-spacing:-.3px; color: #f1f5f9; }}
.brand-sub {{ color:#94a3b8; font-size:12px; margin-bottom:6px; }}

.hero {{ background: linear-gradient(135deg, rgba(30,41,59,.55), rgba(15,23,42,.35));
    border:1px solid rgba(255,255,255,.08); border-radius:16px; padding:22px 26px; margin-bottom:18px;
    backdrop-filter: blur(10px); }}
.hero h1 {{ margin:6px 0 4px; font-size:28px; font-weight:800; letter-spacing:-.6px;
    background: linear-gradient(90deg,#f8fafc,#7dd3fc 60%,#a5b4fc); -webkit-background-clip:text;
    -webkit-text-fill-color:transparent; }}
.hero p {{ margin:0; color:#94a3b8; font-size:14px; }}
.badge {{ display:inline-flex; align-items:center; gap:6px; font-size:11.5px; font-weight:600; padding:3px 10px;
    border-radius:999px; background:rgba(16,185,129,.10); border:1px solid rgba(16,185,129,.30); color:#34d399; }}
.badge.warn {{ background:rgba(245,158,11,.10); border-color:rgba(245,158,11,.35); color:#fbbf24; }}
.dot {{ width:6px; height:6px; border-radius:50%; background:currentColor; box-shadow:0 0 8px currentColor; }}

.kpi {{ background: rgba(30,41,59,.40); border:1px solid rgba(255,255,255,.08); border-radius:14px;
    padding:14px 16px; height:100%; transition: transform .2s ease, border-color .2s ease; }}
.kpi:hover {{ transform: translateY(-2px); border-color: rgba(56,189,248,.45); }}
.kpi-l {{ color:#94a3b8; font-size:11.5px; font-weight:600; text-transform:uppercase; letter-spacing:.6px; }}
.kpi-v {{ color:#f8fafc; font-size:22px; font-weight:700; margin-top:4px; letter-spacing:-.3px; }}
.kpi-s {{ color:#64748b; font-size:12px; margin-top:2px; }}

.stButton>button, .stDownloadButton>button {{ border-radius:10px !important; font-weight:600 !important;
    border:1px solid rgba(255,255,255,.12) !important; transition: all .2s ease !important; }}
.stButton>button:hover, .stDownloadButton>button:hover {{ border-color:#38bdf8 !important;
    box-shadow:0 6px 20px rgba(56,189,248,.25); transform: translateY(-1px); }}

div[data-testid="stChatMessage"] {{ background: {chat_bg} !important; border: 1px solid {chat_border} !important;
    border-radius:14px; padding:12px 16px; direction: {text_dir} !important; text-align: {text_align} !important; }}
div[data-testid="stChatInput"] {{ direction: {text_dir} !important; text-align: {text_align} !important; }}

button[data-baseweb="tab"] {{ font-weight:600; }}
div[data-testid="stMetricValue"] {{ color:#38bdf8; }}
.stMarkdown, p, h1, h2, h3, h4 {{ color: {text_color} !important; }}
</style>
""", unsafe_allow_html=True)

st.session_state.setdefault("model_version", 0)
st.session_state.setdefault("messages", [])

# =====================================================================
# Models and simulation
# =====================================================================
def model_path(domain):
    return os.path.join(MODELS_DIR, f"{domain}.pth")

def is_trained(domain):
    return os.path.exists(model_path(domain))

@st.cache_resource
def load_model(domain, version):
    if not is_trained(domain):
        return None
    model = PhysicsEngineCore(domain)
    model.load_state_dict(torch.load(model_path(domain), map_location="cpu"))
    model.eval()
    return model

def loss_history(domain):
    path = os.path.join(MODELS_DIR, f"{domain}_loss.npy")
    return np.load(path) if os.path.exists(path) else None

@st.cache_data(show_spinner=False)
def simulate(system, alpha, horizon, version):
    model = load_model(system, version)
    if model is None:
        raise RuntimeError(f"No trained model for '{system}'. Open the Model Lab page and train it first.")
    t = np.linspace(0, horizon, 400)
    y_pinn = model.predict(t, alpha)
    y_ref = reference_solution(system, t, alpha)
    _, res_loss = model.solve_ode(torch.tensor(t, dtype=torch.float32).reshape(-1, 1), alpha=alpha)
    err = np.abs(y_pinn - y_ref)
    return {
        "system": system, "label": DOMAINS[system]["label"],
        "alpha": float(alpha), "horizon": float(horizon),
        "residual_loss": float(res_loss.item()),
        "mean_abs_error": float(err.mean()), "max_abs_error": float(err.max()),
        "final_value_pinn": float(y_pinn[-1]), "final_value_reference": float(y_ref[-1]),
        "summary": summarize(system, alpha, t, y_pinn),
        "t": t, "y_pinn": y_pinn, "y_ref": y_ref,
    }

def run_simulation(system, alpha, horizon):
    return simulate(system, round(float(alpha), 2), round(float(horizon), 1), st.session_state.model_version)

@st.cache_data(show_spinner=False)
def error_grid(domain, version):
    model = load_model(domain, version)
    alphas = np.round(np.linspace(0, ALPHA_MAX, 9), 2)
    horizons = [5, 10, 15, 20]
    z = []
    for h in horizons:
        t = np.linspace(0, h, 200)
        z.append([float(np.mean(np.abs(model.predict(t, a) - reference_solution(domain, t, a)))) for a in alphas])
    return alphas, horizons, np.array(z)

# =====================================================================
# Charts (Plotly, interactive)
# =====================================================================
def themed(fig, title=None, height=380, xt="t (s)", yt="y(t)"):
    grid = "rgba(148,163,184,0.12)"
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=8, r=8, t=52 if title else 18, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,17,23,0.55)",
        title=dict(text=title, font=dict(size=15)) if title else None,
        legend=dict(orientation="h", y=1.12, x=0),
        font=dict(family="Inter, sans-serif", color="#cbd5e1"),
        xaxis=dict(title=xt, gridcolor=grid, zeroline=False),
        yaxis=dict(title=yt, gridcolor=grid, zeroline=False), hovermode="x unified")
    return fig

def fig_trajectory(res):
    f = go.Figure()
    f.add_scatter(x=res["t"], y=res["y_ref"], name="Numerical reference (SciPy)",
                  line=dict(color=COL["ref"], width=2, dash="dash"))
    f.add_scatter(x=res["t"], y=res["y_pinn"], name="PINN prediction",
                  line=dict(color=COL["pinn"], width=3))
    return themed(f, f"{res['label']} · alpha={res['alpha']} · T={res['horizon']} s")

def fig_phase(res):
    f = go.Figure()
    for name, y, c, d in [("Reference", res["y_ref"], COL["ref"], "dash"), ("PINN", res["y_pinn"], COL["pinn"], "solid")]:
        f.add_scatter(x=y, y=velocity(res["t"], y), name=name, mode="lines", line=dict(color=c, width=2.4, dash=d))
    f.add_scatter(x=[res["y_pinn"][0]], y=[velocity(res["t"], res["y_pinn"])[0]], mode="markers", name="start",
                  marker=dict(color=COL["amber"], size=10))
    f.update_layout(hovermode="closest")
    return themed(f, "Phase portrait (position vs velocity)", xt="y", yt="dy/dt")

def fig_energy(res):
    f = go.Figure()
    if res["system"] == "thermal":
        f.add_scatter(x=res["t"], y=np.abs(res["y_pinn"]) + 1e-12, name="PINN |y|", line=dict(color=COL["pinn"], width=3))
        f.add_scatter(x=res["t"], y=np.abs(res["y_ref"]) + 1e-12, name="Reference |y|",
                      line=dict(color=COL["ref"], width=2, dash="dash"))
        f.update_yaxes(type="log")
        return themed(f, "Exponential decay (log scale)", yt="|temperature difference|")
    f.add_scatter(x=res["t"], y=energy(res["system"], res["t"], res["y_ref"]), name="Reference",
                  line=dict(color=COL["ref"], width=2, dash="dash"))
    f.add_scatter(x=res["t"], y=energy(res["system"], res["t"], res["y_pinn"]), name="PINN",
                  line=dict(color=COL["violet"], width=3))
    return themed(f, "Mechanical energy over time", yt="E(t)")

def fig_error(res):
    f = go.Figure()
    f.add_scatter(x=res["t"], y=np.abs(res["y_pinn"] - res["y_ref"]), name="|PINN - reference|",
                  line=dict(color=COL["amber"], width=2.5), fill="tozeroy", fillcolor="rgba(245,158,11,0.12)")
    return themed(f, "Absolute error over time", yt="|error|")

def fig_loss(domain):
    hist = loss_history(domain)
    if hist is None:
        return None
    f = go.Figure()
    f.add_scatter(x=np.arange(1, len(hist) + 1), y=hist, name="physics residual loss",
                  line=dict(color=COL["amber"], width=1.6))
    f.update_yaxes(type="log")
    return themed(f, "Real training loss (log scale)", xt="epoch", yt="loss")

# =====================================================================
# Reusable UI pieces
# =====================================================================
def kpi(label, value, sub=""):
    return (f"<div class='kpi'><div class='kpi-l'>{label}</div>"
            f"<div class='kpi-v'>{value}</div><div class='kpi-s'>{sub}</div></div>")

def fmt(value, unit="", digits=2, none="n/a"):
    return none if value is None else f"{value:.{digits}f}{unit}"

def hero(title, subtitle):
    n_ready = sum(is_trained(d) for d in DOMAINS)
    cls, text = ("badge", f"{n_ready}/{len(DOMAINS)} models online") if n_ready == len(DOMAINS) \
        else ("badge warn", f"{n_ready}/{len(DOMAINS)} models trained")
    st.markdown(f"<div class='hero'><span class='{cls}'><span class='dot'></span>{text}</span>"
                f"<h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)

def build_pdf(res):
    s = res["summary"]
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.1, 0.92, "PhysiChat - Simulation Report", fontsize=18, weight="bold")
        lines = [
            f"System: {res['label']}",
            f"Damping coefficient (alpha): {res['alpha']}",
            f"Time horizon: {res['horizon']} s",
            f"Regime: {s['regime']} ({s['regime_note']})",
            f"Settling time: {fmt(s['settling_time_s'], ' s')}",
            f"Oscillation period: {fmt(s['period_s'], ' s')}",
            f"PINN physics residual (mean squared): {res['residual_loss']:.3e}",
            f"Mean absolute error vs reference: {res['mean_abs_error']:.4f}",
            f"Max absolute error vs reference: {res['max_abs_error']:.4f}",
        ]
        for i, line in enumerate(lines):
            fig.text(0.1, 0.86 - i * 0.035, line, fontsize=11)
        pdf.savefig(fig)
        plt.close(fig)
        fig2, ax = plt.subplots(figsize=(8.27, 5))
        ax.plot(res["t"], res["y_ref"], "--", color="green", label="Reference (SciPy)")
        ax.plot(res["t"], res["y_pinn"], color="blue", label="PINN")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("y(t)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        pdf.savefig(fig2)
        plt.close(fig2)
    return buf.getvalue()

def render_result(res, key):
    s = res["summary"]
    c = st.columns(4)
    c[0].markdown(kpi("Regime", s["regime"], s["regime_note"]), unsafe_allow_html=True)
    c[1].markdown(kpi("Settling time", fmt(s["settling_time_s"], " s"),
                      "time to stay within 2% of peak" if s["settling_time_s"] is not None else "not reached / not applicable"),
                  unsafe_allow_html=True)
    c[2].markdown(kpi("Period", fmt(s["period_s"], " s"), "mean time between peaks"), unsafe_allow_html=True)
    c[3].markdown(kpi("Mean error vs SciPy", f"{res['mean_abs_error']:.4f}", f"max {res['max_abs_error']:.4f}"),
                  unsafe_allow_html=True)
    st.write("")
    tabs = st.tabs(["📈 Trajectory", "🌀 Phase portrait", "⚡ Energy", "🎯 Error", "🧠 Training"])
    with tabs[0]:
        st.plotly_chart(fig_trajectory(res), key=f"{key}_traj")
    with tabs[1]:
        st.plotly_chart(fig_phase(res), key=f"{key}_phase")
    with tabs[2]:
        st.plotly_chart(fig_energy(res), key=f"{key}_energy")
    with tabs[3]:
        st.plotly_chart(fig_error(res), key=f"{key}_err")
    with tabs[4]:
        f = fig_loss(res["system"])
        if f is None:
            st.info("No loss history saved for this model yet.")
        else:
            st.plotly_chart(f, key=f"{key}_loss")
        st.caption(f"Physics residual at these points: {res['residual_loss']:.3e}")
    d1, d2, _ = st.columns([1, 1, 2])
    df = pd.DataFrame({"t": res["t"], "reference": res["y_ref"], "pinn": res["y_pinn"]})
    d1.download_button("⬇ Export CSV", df.to_csv(index=False).encode("utf-8"),
                       "physichat_simulation.csv", "text/csv", key=f"{key}_csv")
    d2.download_button("⬇ PDF report", build_pdf(res), "physichat_report.pdf",
                       "application/pdf", key=f"{key}_pdf")

# =====================================================================
# Database Helpers & Pro Logic
# =====================================================================
def get_user_status(visitor_id: str):
    """جلب حالة المستخدم وحالة Pro من قاعدة بيانات Supabase"""
    try:
        res = supabase.table("visitor_limits").select("*").eq("visitor_id", visitor_id).execute()
        if res.data and len(res.data) > 0:
            user_data = res.data[0]
            return {
                "count": user_data.get("message_count", 0),
                "is_pro": user_data.get("is_pro", False)
            }
        else:
            supabase.table("visitor_limits").insert({
                "visitor_id": visitor_id, 
                "message_count": 0, 
                "is_pro": False
            }).execute()
            return {"count": 0, "is_pro": False}
    except Exception:
        return {"count": 0, "is_pro": False}

if "visitor_id" not in st.session_state:
    st.session_state.visitor_id = str(uuid.uuid4())[:8].upper()

current_visitor_id = st.session_state.visitor_id
user_status = get_user_status(current_visitor_id)
is_pro_user = user_status["is_pro"]
msg_count = user_status["count"]

# =====================================================================
# Pages (General AI + Physics)
# =====================================================================
SUGGESTIONS = [
    "كيفاش نكتب كود Python ساهل لفتح ملفات CSV؟",
    "شنو يوقع لنواس تخميدو 0.8 فـ 15 ثانية؟",
    "شرح ليا مفهوم الذكاء الاصطناعي بالدارجة المغربية",
    "Simulate a pendulum with damping 1.2",
]

def page_chat():
    hero("General AI Copilot",
         "Ask me anything! Programming, science, general knowledge, or request a live physics simulation in Darija, Arabic, or English.")
    pending = st.session_state.pop("pending", None)
    prompt = pending if pending else None

    if not st.session_state.messages:
        st.markdown("**Try one of these:**")
        cols = st.columns(len(SUGGESTIONS))
        for i, (col, text) in enumerate(zip(cols, SUGGESTIONS)):
            if col.button(text, key=f"sugg_btn_{i}_{abs(hash(text))}"):
                st.session_state.pending = text
                st.rerun()

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for j, res in enumerate(msg.get("results", [])):
                render_result(res, f"chat_{i}_{j}")
                
    chat_input_prompt = st.chat_input("سولني على أي حاجة بغيتي (برمجة، فيزياء، أسئلة عامة)")
    if chat_input_prompt:
        prompt = chat_input_prompt

    if prompt:
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        if not is_pro_user and msg_count >= 10:
            st.error("🛑 وصلتي للحد الأقصى اليومي (10 رسائل مجانية).")
            st.info("💡 اشترك في باقة Pro لتستمتع برسائل ومحاكاة غير محدودة!")
            
            whatsapp_number = "212717817730"
            wa_text = f"السلام عليكم، بغيت نفعل حساب PhysiChat Pro. كود الحساب ديالي هو: {current_visitor_id}"
            wa_link = f"https://wa.me/{whatsapp_number}?text={wa_text.replace(' ', '%20')}"
            st.markdown(f'👉 [📲 **اضغط هنا للتواصل والتفعيل عبر WhatsApp**]({wa_link})')
        else:
            if not is_pro_user:
                new_count = msg_count + 1
                try:
                    res_update = supabase.table("visitor_limits").update({
                        "message_count": new_count,
                        "last_date": today_str
                    }).eq("visitor_id", current_visitor_id).execute()
                    
                    if not res_update.data:
                        supabase.table("visitor_limits").insert({
                            "visitor_id": current_visitor_id,
                            "message_count": new_count,
                            "last_date": today_str,
                            "is_pro": False
                        }).execute()
                except Exception as e:
                    print(f"Supabase error: {e}")
                    
                st.caption(f"📊 باقي ليك {10 - new_count} رسائل مجانية اليوم.")
            else:
                st.caption("👑 حساب Pro مفعل - استخدام غير محدود!")
            
            st.session_state.messages.append({"role": "user", "content": prompt, "results": []})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                with st.spinner("جاري التفكير والإجابة..."):
                    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                    try:
                        text, results = chat_turn(history, run_simulation)
                    except Exception as exc:
                        text, results = f"حدث خطأ ما: {exc}", []
                    st.markdown(text)
                    
            st.session_state.messages.append({"role": "assistant", "content": text, "results": results})
            n = len(st.session_state.messages) - 1
            for j, res in enumerate(results):
                render_result(res, f"chat_{n}_{j}")

def page_playground():
    hero("Playground", "Move the sliders and watch the PINN respond instantly. Exclusive for Pro users.")
    
    if not is_pro_user:
        st.error("🔒 ميزة المحاكاة البصرية والمنحنيات حصرية لمشتركي PhysiChat Pro!")
        st.info("💡 اشترك الآن لتفتح جميع الأدوات والمحاكاة التفاعلية بدون حدود.")
        whatsapp_number = "212717817730"
        wa_text = f"السلام عليكم، بغيت نفعل حساب PhysiChat Pro باش نفتح المحاكاة. كود الحساب ديالي هو: {current_visitor_id}"
        wa_link = f"https://wa.me/{whatsapp_number}?text={wa_text.replace(' ', '%20')}"
        st.markdown(f'👉 [🚀 **تواصل معنا عبر WhatsApp لتفعيل Pro**]({wa_link})')
        return

    c1, c2, c3 = st.columns([2, 2, 2])
    domain = c1.selectbox("System", list(DOMAINS), format_func=lambda d: DOMAINS[d]["label"])
    alpha = c2.slider("Damping coefficient (alpha)", 0.0, float(ALPHA_MAX), 0.5, 0.05)
    horizon = c3.slider("Time horizon T (s)", 5.0, 20.0, 10.0, 1.0)
    try:
        render_result(run_simulation(domain, alpha, horizon), "play")
    except RuntimeError as exc:
        st.warning(str(exc))

def page_compare():
    hero("Compare", "Overlay several damping values on one chart. Exclusive for Pro users.")
    
    if not is_pro_user:
        st.error("🔒 ميزة المقارنة والمحاكاة حصرية لمشتركي PhysiChat Pro!")
        whatsapp_number = "212717817730"
        wa_text = f"السلام عليكم، بغيت نفعل حساب PhysiChat Pro. كود الحساب ديالي هو: {current_visitor_id}"
        wa_link = f"https://wa.me/{whatsapp_number}?text={wa_text.replace(' ', '%20')}"
        st.markdown(f'👉 [🚀 **تواصل معنا عبر WhatsApp لتفعيل Pro**]({wa_link})')
        return

    c1, c2, c3 = st.columns([2, 3, 2])
    domain = c1.selectbox("System", list(DOMAINS), format_func=lambda d: DOMAINS[d]["label"], key="cmp_dom")
    alphas = c2.multiselect("Alpha values", [0.0, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0], default=[0.2, 0.8, 1.5])
    horizon = c3.slider("Time horizon T (s)", 5.0, 20.0, 10.0, 1.0, key="cmp_T")
    if not alphas:
        st.info("Pick at least one alpha value.")
        return
    try:
        results = [run_simulation(domain, a, horizon) for a in sorted(alphas)]
    except RuntimeError as exc:
        st.warning(str(exc))
        return
    f = go.Figure()
    for i, r in enumerate(results):
        color = PALETTE[i % len(PALETTE)]
        f.add_scatter(x=r["t"], y=r["y_pinn"], name=f"PINN α={r['alpha']}", line=dict(color=color, width=3))
        f.add_scatter(x=r["t"], y=r["y_ref"], name=f"ref α={r['alpha']}", line=dict(color=color, width=1.5, dash="dot"),
                      showlegend=False)
    st.plotly_chart(themed(f, f"{DOMAINS[domain]['label']}: effect of damping", height=440), key="cmp_fig")
    table = pd.DataFrame([{
        "alpha": r["alpha"], "regime": r["summary"]["regime"],
        "settling time (s)": r["summary"]["settling_time_s"], "period (s)": r["summary"]["period_s"],
        "mean error": round(r["mean_abs_error"], 5),
    } for r in results])
    st.dataframe(table, hide_index=True)

def page_lab():
    hero("Model Lab", "Check how accurate each PINN really is, and train or retrain models.")
    st.markdown("#### Train")
    c1, c2, c3 = st.columns([2, 2, 1])
    to_train = c1.selectbox("Model", list(DOMAINS), format_func=lambda d: DOMAINS[d]["label"], key="train_dom")
    epochs = c2.slider("Epochs", 1000, 10000, 6000, 500)
    c3.write("")
    c3.write("")
    if c3.button("Train", type="primary"):
        with st.spinner(f"Training {DOMAINS[to_train]['label']}... (a few minutes on CPU)"):
            train_domain(to_train, epochs=epochs, verbose=False)
            st.session_state.model_version += 1
        st.success("Training finished.")
        st.rerun()

    st.markdown("#### Accuracy")
    trained = [d for d in DOMAINS if is_trained(d)]
    if not trained:
        st.info("No trained models yet. Train one above, or run `python physics_trainer.py`.")
        return
    domain = st.selectbox("Inspect model", trained, format_func=lambda d: DOMAINS[d]["label"], key="lab_dom")
    hist = loss_history(domain)
    m1, m2, m3 = st.columns(3)
    m1.markdown(kpi("Epochs trained", f"{len(hist):,}" if hist is not None else "n/a"), unsafe_allow_html=True)
    m2.markdown(kpi("Final training loss", f"{hist[-1]:.2e}" if hist is not None else "n/a"), unsafe_allow_html=True)
    m3.markdown(kpi("Input range", f"alpha 0-{ALPHA_MAX:g}, T up to 20 s", "outside this range = unreliable"),
                unsafe_allow_html=True)
    st.write("")
    alphas, horizons, z = error_grid(domain, st.session_state.model_version)
    f = go.Figure(go.Heatmap(z=z, x=[str(a) for a in alphas], y=[f"{h} s" for h in horizons],
                             colorscale="Turbo", colorbar=dict(title="mean |error|"),
                             hovertemplate="alpha=%{x}<br>T=%{y}<br>error=%{z:.4f}<extra></extra>"))
    f.update_layout(hovermode="closest")
    st.plotly_chart(themed(f, "Where is the PINN accurate? (error vs SciPy)", height=340,
                           xt="damping alpha", yt="time horizon"), key="lab_heat")
    lf = fig_loss(domain)
    if lf is not None:
        st.plotly_chart(lf, key="lab_loss")

def page_pricing():
    st.header("👑 باقات PhysiChat Pro")
    st.write("استمتع بذكاء اصطناعي غير محدود، محاكاة بصرية، ورسوم بيانية دقيقة لحل المسائل العلمية!")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("🆓 المجانية")
        st.write("• 10 رسائل مجانية يومياً")
        st.write("• إجابات ذكية وسريعة")
        st.write("• تتجدد تلقائياً كل 24 ساعة")
        st.button("الحالية", disabled=True, key="free_btn")

    with col2:
        st.subheader("🥉 شهري")
        st.markdown("**49 درهم / شهر**")
        st.write("• رسائل غير محدودة ♾️")
        st.write("• محاكاة بصرية كاملة")
        st.button("اختر الشهري", key="m_btn")

    with col3:
        st.subheader("🥈 6 أشهر")
        st.markdown("**125 درهم / 6 أشهر**")
        st.write("• توفير مهم للفصل الدراسي")
        st.write("• محاكاة بصرية كاملة")
        st.button("اختر 6 أشهر", key="s_btn")

    with col4:
        st.subheader("👑 سنوي")
        st.markdown("**200 درهم / سنة**")
        st.write("• توفير كبير جداً ⭐")
        st.write("• دعم ومميزات حصرية")
        st.button("اختر السنوي", key="y_btn")

    st.markdown("---")
    st.markdown("📲 **طريقة التفعيل الفوري عبر WhatsApp:**")
    st.info(f"كود الحساب الخاص بك هو: **{current_visitor_id}** (قم بنسخه وإرساله عبر الواتساب لتفعيل باقتك فوراً)")
    
    whatsapp_number = "212717817730" 
    wa_text = f"السلام عليكم، بغيت نفعل باقة Pro. كود الحساب ديالي هو: {current_visitor_id}"
    wa_link = f"https://wa.me/{whatsapp_number}?text={wa_text.replace(' ', '%20')}"
    
    st.markdown(f'👉 [📲 **اضغط هنا للتواصل والتفعيل عبر WhatsApp**]({wa_link})')

# =====================================================================
# Sidebar & Navigation
# =====================================================================
with st.sidebar:
    st.markdown(f"<div class='brand'><div class='brand-logo'>⚛️</div><div class='brand-name'>PhysiChat</div></div>"
                f"<div class='brand-sub'>كود الحساب: <b>{current_visitor_id}</b></div>", unsafe_allow_html=True)
    
    page = st.radio(
        "القائمة الرئيسية", 
        [
            "💬 Copilot Chat", 
            "⚅ Playground", 
            "⚖️ Compare", 
            "🧪 Model Lab", 
            "👑 باقات PhysiChat Pro"
        ],
        label_visibility="collapsed"
    )
    
    st.divider()
    st.markdown("**حالة النماذج (Models)**")
    for d, cfg in DOMAINS.items():
        st.write("🟢 " + cfg["label"])
        
    st.divider()
    if is_pro_user:
        st.success("👑 حساب Pro مفعل بنجاح")
    else:
        st.warning(f"📊 الرسائل اليومية: {msg_count}/10")
        
    st.caption("AI Engine: Groq Llama 3 (Active)")
    if st.button("🗑 Clear chat"):
        st.session_state.messages = []
        st.rerun()

# =====================================================================
# Main Page Router
# =====================================================================
pages = {
    "💬 Copilot Chat": page_chat,
    "⚅ Playground": page_playground,
    "⚖️ Compare": page_compare,
    "🧪 Model Lab": page_lab,
    "👑 باقات PhysiChat Pro": page_pricing
}

pages[page]()
