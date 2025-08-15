from __future__ import annotations
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load local .env during development
load_dotenv()

def _read_api_key() -> str:
    # Streamlit secrets if available (works both locally and on Streamlit Cloud)
    try:
        import streamlit as st  # optional import
        key = st.secrets.get("GOOGLE_API_KEY", None)
        if key:
            return key
    except Exception:
        pass

    # Environment variables
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "Gemini API key not found. Set GOOGLE_API_KEY (preferred) or GEMINI_API_KEY."
        )
    return key

# Singleton client (cached by module import)
API_KEY = _read_api_key()
client = genai.Client(api_key=API_KEY)

# Default model: fast & cost-efficient; you can change to 'gemini-2.5-pro' for quality
DEFAULT_MODEL = "gemini-1.5-flash"

# Optional: safety settings (be conservative on blocks)
DEFAULT_SAFETY = [
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH"),
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_ONLY_HIGH"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_ONLY_HIGH"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH"),
]
