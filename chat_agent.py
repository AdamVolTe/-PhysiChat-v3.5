"""
PhysiChat agent (Groq):
- General chat + tool calling (physics simulation, web search, symbolic math)
- The API key is read lazily (st.secrets -> env/.env), so the app never crashes at import
- Simulation tool results carry the REAL numbers, so the model explains actual results
- Offline fallback: a keyword parser that still runs simulations without any API key
"""
import json
import os
import re

import streamlit as st
from dotenv import load_dotenv

from math_engine import simplify_expression
from physics_engine import ALPHA_MAX, DOMAINS, T_MAX
from web_search import search_web

load_dotenv()

# Override with GROQ_MODEL in .env. See https://console.groq.com/docs/models for the available model IDs.
MODEL_ID = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
MIN_HORIZON = 5.0

SYSTEM_PROMPT = (
    "You are PhysiChat AI, a powerful, versatile and friendly general AI assistant created by Adam. "
    "You can answer questions on ANY topic: programming, science, general knowledge, writing, languages. "
    "You understand and speak Moroccan Darija, Arabic, English and French naturally; reply in the user's language.\n\n"
    "INSTRUCTIONS:\n"
    "1. For general questions, answer directly with helpful, accurate, friendly responses.\n"
    "2. For physical systems / simulations (pendulum, oscillator, cooling, driven motion) call `run_simulation`, "
    "then explain the result using ONLY the numbers returned by the tool. Never invent numbers. "
    "If `inputs_were_clamped` is true, tell the user their values were adjusted to the trained range.\n"
    "3. For real-time info or current events call `web_search_tool`.\n"
    "4. For exact symbolic math call `math_solve_tool`."
)

TOOLS = [
    {"type": "function", "function": {
        "name": "run_simulation",
        "description": "Run a PINN physics simulation (compared against SciPy) for a dynamical system.",
        "parameters": {"type": "object", "properties": {
            "system": {"type": "string", "enum": list(DOMAINS)},
            "alpha": {"type": "number", "description": f"Damping coefficient (0.0 to {ALPHA_MAX:g})"},
            "horizon": {"type": "number", "description": f"Time horizon in seconds ({MIN_HORIZON:g} to {T_MAX:g})"},
        }, "required": ["system"]}}},
    {"type": "function", "function": {
        "name": "web_search_tool",
        "description": "Live web search for recent news, papers or real-time info.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Search query terms"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "math_solve_tool",
        "description": "Simplify a math expression symbolically with SymPy.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "Mathematical expression to simplify"}},
            "required": ["expression"]}}},
]


# ---------------------------------------------------------------- API key / client
def get_api_key():
    """Returns the Groq key (Streamlit secrets first, then env/.env) or None."""
    try:
        key = st.secrets["GROQ_API_KEY"]
    except Exception:
        key = None
    return key or os.getenv("GROQ_API_KEY")


_client = None


def _get_client():
    global _client
    if _client is None:
        key = get_api_key()
        if not key:
            return None
        from groq import Groq
        _client = Groq(api_key=key)
    return _client


# ---------------------------------------------------------------- simulation helpers
def _to_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_sim_args(raw):
    """Validates and clamps tool arguments to the range the PINNs were trained on."""
    system = raw.get("system", "oscillator")
    if system not in DOMAINS:
        system = "oscillator"
    alpha_in = _to_float(raw.get("alpha"), 0.5)
    horizon_in = _to_float(raw.get("horizon"), 10.0)
    alpha = min(max(alpha_in, 0.0), ALPHA_MAX)
    horizon = min(max(horizon_in, MIN_HORIZON), T_MAX)
    clamped = (alpha != alpha_in) or (horizon != horizon_in)
    return {"system": system, "alpha": alpha, "horizon": horizon}, clamped


def _simulation_payload(res, clamped):
    """Compact JSON with the real numbers, so the LLM can explain the actual result."""
    s = res["summary"]
    r = lambda v, d=5: None if v is None else round(float(v), d)
    return json.dumps({
        "system": res["system"], "label": res["label"],
        "alpha": res["alpha"], "horizon_s": res["horizon"],
        "regime": s["regime"], "regime_note": s["regime_note"],
        "peak_abs": r(s["peak_abs"]), "settling_time_s": r(s["settling_time_s"], 2),
        "period_s": r(s["period_s"], 3),
        "final_value_pinn": r(res["final_value_pinn"]), "final_value_scipy": r(res["final_value_reference"]),
        "mean_abs_error_vs_scipy": r(res["mean_abs_error"]), "max_abs_error_vs_scipy": r(res["max_abs_error"]),
        "physics_residual": f"{res['residual_loss']:.2e}",
        "inputs_were_clamped": clamped,
        "trained_range": {"alpha": [0, ALPHA_MAX], "horizon_s": [MIN_HORIZON, T_MAX]},
    }, ensure_ascii=False)


def _run_tool(name, args, run_simulation_fn, results):
    try:
        if name == "run_simulation":
            clean, clamped = _clean_sim_args(args)
            res = run_simulation_fn(**clean)
            results.append(res)
            return _simulation_payload(res, clamped)
        if name == "web_search_tool":
            return search_web(str(args.get("query", ""))[:300])
        if name == "math_solve_tool":
            return json.dumps(simplify_expression(str(args.get("expression", ""))), ensure_ascii=False)
        return "Unknown tool."
    except Exception as e:
        return f"Tool error: {e}"


# ---------------------------------------------------------------- offline fallback
_SYSTEM_WORDS = [
    ("pendulum", ("pendulum", "pendule", "نواس", "بندول")),
    ("thermal", ("cooling", "thermal", "refroid", "تبريد", "حرار")),
    ("driven", ("driven", "forced", "forcé", "قسري", "مجبر")),
    ("oscillator", ("oscillat", "oscillateur", "تذبذب", "اهتزاز", "محاكاة", "simulat", "تخميد", "damping", "amortissement")),
]
_ALPHA_RE = re.compile(r"(?:alpha|α|damping|amortissement|تخميد\w*)\D{0,6}(\d+(?:[.,]\d+)?)", re.I)
_TIME_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:s\b|sec|seconds?|secondes?|ثانية|ثواني|ثانيه)", re.I)


def _offline_reply(text, run_simulation_fn):
    low = text.lower()
    system = next((name for name, words in _SYSTEM_WORDS if any(w in low for w in words)), None)
    if system is None:
        return ("⚠️ ما كاينش مفتاح GROQ_API_KEY، فالـ AI العام ما خدامش دابا. "
                "ولكن نقدر نشغل ليك محاكاة: جرب «نواس تخميدو 0.8 فـ 15 ثانية».", [])
    m_a, m_t = _ALPHA_RE.search(text), _TIME_RE.search(text)
    raw = {"system": system,
           "alpha": m_a.group(1).replace(",", ".") if m_a else 0.5,
           "horizon": m_t.group(1).replace(",", ".") if m_t else 10.0}
    clean, _ = _clean_sim_args(raw)
    res = run_simulation_fn(**clean)
    s = res["summary"]
    period = f"{s['period_s']:.2f} s" if s["period_s"] else "—"
    settle = f"{s['settling_time_s']:.2f} s" if s["settling_time_s"] is not None else "—"
    msg = (f"(وضع بلا AI) **{res['label']}** · alpha={res['alpha']} · T={res['horizon']} s\n\n"
           f"- النظام: **{s['regime']}** — {s['regime_note']}\n"
           f"- زمن الاستقرار: {settle} · الدور: {period}\n"
           f"- الخطأ المتوسط مقارنة بـ SciPy: {res['mean_abs_error']:.4f}")
    return msg, [res]


# ---------------------------------------------------------------- main entry point
def chat_turn(history, run_simulation_fn, max_tool_rounds=3):
    """history: [{"role","content"}, ...]. Returns (assistant_text, [simulation results])."""
    client = _get_client()
    if client is None:
        return _offline_reply(history[-1]["content"], run_simulation_fn)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
        {"role": m["role"], "content": m["content"]} for m in history]
    results = []
    try:
        for round_no in range(max_tool_rounds + 1):
            kwargs = dict(model=MODEL_ID, messages=messages, temperature=0.7, max_tokens=2048)
            if round_no < max_tool_rounds:            # last round: force a final text answer
                kwargs.update(tools=TOOLS, tool_choice="auto")
            msg = client.chat.completions.create(**kwargs).choices[0].message
            if not msg.tool_calls:
                return msg.content or "", results

            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.function.name,
                                 "content": _run_tool(tc.function.name, args, run_simulation_fn, results)})
        return "", results
    except Exception as e:
        return f"عذراً، وقع خطأ أثناء معالجة الطلب: {e}", results
