import os
import sys
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("[ERROR] GEMINI_API_KEY not found in .env file.")
    sys.exit(1)

genai.configure(api_key=api_key)

print("\n" + "=" * 50)
print("  Available Gemini Models (generateContent)")
print("=" * 50)

try:
    models = [
        m for m in genai.list_models()
        if "generateContent" in m.supported_generation_methods
    ]

    if not models:
        print("  No models found. Check your API key permissions.")
    else:
        for m in models:
            print(f"  ✓ {m.name}")

except Exception as e:
    print(f"[ERROR] Failed to fetch models: {e}")
    sys.exit(1)

print("=" * 50 + "\n")