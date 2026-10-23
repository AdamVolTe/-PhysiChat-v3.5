"""
Enhanced General AI Chat agent (Groq Edition):
- General purpose AI (coding, languages, general knowledge, science)
- Fast and free Groq Llama 3 / Llama 3.3 integration
- Full tool calling (Web Search, Math Engine, Physics Simulations)
- Secure API key management via environment variables
"""
import json
import os
import re
import streamlit as st
from web_search import search_web
from math_engine import simplify_expression
from groq import Groq
from dotenv import load_dotenv

# تحضير وتعيين المتغيرات البيئية من ملف .env
load_dotenv()

# قراءة المفتاح بأمان سواء محلياً أو على Streamlit Cloud

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("⚠️ لم يتم العثور على GROQ_API_KEY لا في Secrets ولا في ملف .env!")

client = Groq(api_key=GROQ_API_KEY)

MODEL_ID = "qwen/qwen3.8-27b"

SYSTEM_PROMPT = (
    "You are PhysiChat AI, a powerful, versatile, and highly intelligent general AI assistant created by Adam. "
    "You can answer questions on ANY topic including programming, science, general knowledge, philosophy, writing, languages, and casual conversation. "
    "You seamlessly understand and speak Moroccan Darija, Arabic, English, French, and other languages naturally.\n\n"
    "CRITICAL INSTRUCTIONS:\n"
    "1. For general inquiries, coding, explanations, or creative requests, provide helpful, detailed, friendly, and accurate responses directly.\n"
    "2. When users specifically ask about physical systems, dynamic simulations, or trajectories (pendulum, oscillator, cooling, driven motion, etc.), call the `run_simulation` tool and explain the results clearly.\n"
    "3. For real-time information, latest news, or current events, call `web_search_tool`.\n"
    "4. For exact mathematical calculations or symbolic equations, call `math_solve_tool`."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_simulation",
            "description": "Run a PINN physics simulation for dynamical systems when requested by the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "system": {"type": "string", "enum": ["oscillator", "pendulum", "thermal", "driven"]},
                    "alpha": {"type": "number", "description": "Damping coefficient (0.0 to 2.0)"},
                    "horizon": {"type": "number", "description": "Time horizon in seconds (5.0 to 20.0)"}
                },
                "required": ["system"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search_tool",
            "description": "Perform a live web search for recent news, papers, or real-time info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query terms"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "math_solve_tool",
            "description": "Simplify a math or physics expression using SymPy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Mathematical expression to simplify"}
                },
                "required": ["expression"]
            }
        }
    }
]

def chat_turn(history, run_simulation_fn, max_tool_rounds=3):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history:
        messages.append({"role": m["role"], "content": m["content"]})

    results = []
    ai_response_text = ""
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=2048,
        )
        response_message = completion.choices[0].message
        ai_response_text = response_message.content or ""

        # التعامل مع Tool Calls في حال استدعاء النموذج للأدوات
        if response_message.tool_calls:
            messages.append(response_message)
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                if function_name == "run_simulation":
                    args = {
                        "system": function_args.get("system", "oscillator"),
                        "alpha": float(function_args.get("alpha", 0.5)),
                        "horizon": float(function_args.get("horizon", 10.0)),
                    }
                    res = run_simulation_fn(**args)
                    results.append(res)
                    tool_response = f"تم تشغيل المحاكاة بنجاح للنظام {res['label']} بمعامل تخميد {args['alpha']}."
                elif function_name == "web_search_tool":
                    tool_response = search_web(function_args.get("query", ""))
                elif function_name == "math_solve_tool":
                    tool_response = str(simplify_expression(function_args.get("expression", "")))
                else:
                    tool_response = "Unknown tool."

                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": tool_response
                })

            # استدعاء ثانٍ للحصول على التفسير النهائي مدعوماً بنتائج الأدوات
            second_completion = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
            )
            ai_response_text = second_completion.choices[0].message.content

    except Exception as e:
        ai_response_text = f"عذراً، حدث خطأ أثناء معالجة الطلب: {str(e)}"

    return ai_response_text, results

def get_api_key():
    """دالة توافقية مطلوبة من app.py"""
    return GROQ_API_KEY
