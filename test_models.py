import os
from groq import Groq

# حط المفتاح ديالك المباشر هنا بين الشرتين ""
api_key = os.getenv("GROQ_API_KEY") or "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"

client = Groq(api_key=api_key)

try:
    print("Fetching active models...")
    models = client.models.list()
    for model in models.data:
        print(f"- {model.id}")
except Exception as e:
    print(f"Error: {e}")
