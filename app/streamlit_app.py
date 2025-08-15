from __future__ import annotations
import io
import hashlib
from typing import List
import streamlit as st
from PIL import Image
from core.captioner import generate_captions, CaptionOption
from core.config import DEFAULT_MODEL

st.set_page_config(page_title="AI Image Caption Recommender", page_icon="🖼️", layout="centered")

st.title("🖼️ AI Image Caption Recommender (Gemini)")
st.caption("Upload an image → get multiple caption suggestions, alt text, and (optional) hashtags.")

with st.sidebar:
    st.header("Settings")
    model = st.selectbox(
        "Gemini model",
        options=["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
        index=["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"].index(DEFAULT_MODEL)
    )
    n = st.slider("Number of captions", 3, 10, 5)
    styles = st.multiselect(
        "Styles to include",
        ["Neutral", "Witty", "Storylike", "Minimal", "Poetic", "Elegant", "Playful", "SEO"],
        default=["Neutral", "Witty", "Storylike", "Minimal"],
    )
    platform = st.selectbox("Platform (optional)", ["", "Instagram", "LinkedIn", "Twitter/X", "Pinterest"])
    include_hashtags = st.checkbox("Include hashtags", value=True)
    max_chars = st.select_slider("Max characters (optional)", options=[0, 80, 120, 150, 220], value=0)
    extra_keywords = st.text_input("Extra keywords (optional)", placeholder="e.g., summer, travel, sunset")

uploaded = st.file_uploader("Upload an image (JPG/PNG/WebP)", type=["jpg", "jpeg", "png", "webp"])

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

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Preview", use_container_width=True)

    if st.button("✨ Generate captions", type="primary"):
        with st.spinner("Calling Gemini..."):
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

        st.subheader("Results")
        for i, opt in enumerate(results, start=1):
            st.markdown(f"**Option {i} · _{opt.style}_**")
            # st.code shows a handy copy button
            st.code(opt.caption.strip())
            with st.expander("Alt text"):
                st.write(opt.alt_text.strip())
            if opt.hashtags:
                st.write("Hashtags:")
                st.code(" ".join(f"#{h.lstrip('#')}" for h in opt.hashtags))
            if opt.score is not None:
                st.caption(f"Model self-score: {opt.score:.1f}/10")

        # Download all as JSON
        import json
        json_payload = json.dumps([opt.model_dump() for opt in results], ensure_ascii=False, indent=2)
        st.download_button("⬇️ Download JSON", data=json_payload, file_name="captions.json", mime="application/json")

else:
    st.info("Drop an image to begin.")
