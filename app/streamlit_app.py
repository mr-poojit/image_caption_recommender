from __future__ import annotations
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import io
import hashlib
import json
from typing import List
import streamlit as st
from PIL import Image
from core.captioner import generate_captions, CaptionOption
from core.config import DEFAULT_MODEL

# ==== PAGE CONFIG ====
st.set_page_config(page_title="AI Image Caption Recommender", page_icon="🎨", layout="wide")

# ==== CUSTOM CSS ====
st.markdown("""
<style>
/* ===== Background Gradient Animation ===== */
body {
  margin: 0;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background: linear-gradient(-45deg, #6a11cb, #2575fc, #b721ff, #21d4fd);
  background-size: 400% 400%;
  animation: gradientBG 12s ease infinite;
  color: white;
}
@keyframes gradientBG {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

/* ===== Image Preview ===== */
.preview-container {
  display: flex;
  justify-content: center;
  margin-top: 15px;
}
.preview-container img {
  width: 200px !important; /* Fixed width */
  height: auto !important; /* Keep aspect ratio */
  border-radius: 15px;
  box-shadow: 0 6px 25px rgba(0, 0, 0, 0.4);
  transition: all 0.3s ease;
}
.preview-container img:hover {
  transform: scale(1.05);
  box-shadow: 0 8px 30px rgba(255, 255, 255, 0.4);
}

/* ===== Caption Card ===== */
.caption-card {
  background: rgba(255, 255, 255, 0.08);
  border-radius: 15px;
  padding: 15px;
  margin: 10px 0;
  backdrop-filter: blur(10px);
  box-shadow: 0 4px 15px rgba(0,0,0,0.25);
  transition: all 0.3s ease;
}
.caption-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 6px 20px rgba(0,0,0,0.4);
}

/* ===== Fancy Button ===== */
.stButton button {
  background: linear-gradient(90deg, #ff6ec4, #7873f5);
  color: white;
  padding: 10px 20px;
  border-radius: 25px;
  border: none;
  font-size: 16px;
  font-weight: bold;
  transition: all 0.3s ease;
}
.stButton button:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

/* ===== Expander styling ===== */
.streamlit-expanderHeader {
  font-weight: bold;
  color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ==== HEADER ====
st.markdown("## ✨ AI Image Caption Recommender")
st.caption("Upload an image → Get creative captions, alt text, and hashtags with a **stylish purple UI**.")

# ==== SIDEBAR ====
with st.sidebar:
    st.header("⚙️ Settings")
    model = st.selectbox(
        "Gemini model",
        options=["gemini-1.5-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
        index=["gemini-1.5-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"].index(DEFAULT_MODEL)
    )
    n = st.slider("Number of captions", 3, 10, 5)
    styles = st.multiselect(
        "Caption Styles",
        ["Neutral", "Witty", "Storylike", "Minimal", "Poetic", "Elegant", "Playful", "SEO"],
        default=["Neutral", "Witty", "Storylike", "Minimal"],
    )
    platform = st.selectbox("Platform", ["", "Instagram", "LinkedIn", "Twitter/X", "Pinterest"])
    include_hashtags = st.checkbox("Include hashtags", value=True)
    max_chars = st.select_slider("Max characters", options=[0, 80, 120, 150, 220], value=0)
    extra_keywords = st.text_input("Extra keywords", placeholder="e.g., summer, travel, sunset")

# ==== IMAGE UPLOAD ====
uploaded = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png", "webp"])

# ==== HELPERS ====
def _hash_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()

@st.cache_data(show_spinner=False)
def _cached_captions(img_bytes: bytes, n: int, styles: List[str], platform: str, include_hashtags: bool,
                     max_chars: int, extra_keywords: str, model: str) -> list[CaptionOption]:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    maxc = None if max_chars == 0 else int(max_chars)
    return generate_captions(
        image=img,
        n=n,
        styles=styles,
        platform=platform if platform else None,
        include_hashtags=include_hashtags,
        max_chars=maxc,
        extra_keywords=extra_keywords if extra_keywords else None,
        model=model,
    )

# ==== MAIN LOGIC ====
if uploaded:
    image = Image.open(uploaded).convert("RGB")

    # Resize to fixed preview size
    fixed_width = 200
    ratio = fixed_width / image.width
    new_height = int(image.height * ratio)
    preview_img = image.resize((fixed_width, new_height))

    # Centered preview container
    st.markdown('<div class="preview-container">', unsafe_allow_html=True)
    st.image(preview_img, caption="Preview", use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("✨ Generate captions"):
        with st.spinner("🎯 Crafting captions with Gemini..."):
            img_bytes = uploaded.getvalue()
            results = _cached_captions(
                img_bytes,
                n,
                styles,
                platform or "",
                include_hashtags,
                max_chars,
                extra_keywords or "",
                model,
            )

        st.markdown("### 🏆 Results")
        for i, opt in enumerate(results, start=1):
            st.markdown(f"""
            <div class="caption-card">
                <h4>Option {i} · <i>{opt.style}</i></h4>
                <code>{opt.caption.strip()}</code>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("📝 Alt text"):
                st.write(opt.alt_text.strip())
            if opt.hashtags:
                st.write("🏷️ Hashtags:")
                st.code(" ".join(f"#{h.lstrip('#')}" for h in opt.hashtags))
            if opt.score is not None:
                st.caption(f"⭐ Model self-score: {opt.score:.1f}/10")

        # JSON Download
        json_payload = json.dumps([opt.model_dump() for opt in results], ensure_ascii=False, indent=2)
        st.download_button("⬇️ Download JSON", data=json_payload, file_name="captions.json", mime="application/json")

else:
    st.info("🚀 Upload an image to get started!")
