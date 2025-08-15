from __future__ import annotations
import json
from typing import Any, List, Dict
from PIL import Image
from pydantic import BaseModel, Field
from google.genai import types
from .config import client, DEFAULT_MODEL, DEFAULT_SAFETY
from .prompts import build_caption_prompt

class CaptionOption(BaseModel):
    caption: str
    style: str
    length: str | None = None
    alt_text: str
    hashtags: List[str] = Field(default_factory=list)
    score: float | None = None  # self-score for relevance/engagement

class CaptionList(BaseModel):
    __root__: List[CaptionOption]

def _schema_array_of_caption_options() -> types.Schema:
    # See "Structured output" in Gemini docs
    return types.Schema(
        type="ARRAY",
        items=types.Schema(
            type="OBJECT",
            properties={
                "caption": types.Schema(type="STRING", description="Human-like image caption."),
                "style": types.Schema(type="STRING", description="Style label, e.g., Witty, Minimal, Story."),
                "length": types.Schema(type="STRING", description="Short/Medium/Long"),
                "alt_text": types.Schema(type="STRING", description="Accessible alt text that describes the image."),
                "hashtags": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
                "score": types.Schema(type="NUMBER", description="0-10 self score for engagement & relevance"),
            },
            required=["caption", "style", "alt_text", "hashtags"],
        )
    )

def generate_captions(
    image: Image.Image,
    n: int = 5,
    styles: list[str] | None = None,
    platform: str | None = None,
    include_hashtags: bool = True,
    max_chars: int | None = None,
    extra_keywords: str | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
) -> list[CaptionOption]:
    """
    Returns parsed list of CaptionOption using Gemini structured output.
    You can pass a PIL Image directly to the SDK (it supports image parts).
    """
    styles = styles or ["Neutral", "Witty", "Storylike", "Minimal"]
    prompt = build_caption_prompt(
        n=n,
        styles=styles,
        platform=platform,
        include_hashtags=include_hashtags,
        max_chars=max_chars,
        extra_keywords=extra_keywords,
    )

    response = client.models.generate_content(
        model=model,
        contents=[prompt, image],   # PIL Image is accepted as an input part
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_schema_array_of_caption_options(),
            temperature=temperature,
            max_output_tokens=1024,
            safety_settings=DEFAULT_SAFETY,
        ),
    )

    # The SDK exposes JSON via response.text when response_mime_type="application/json"
    raw = response.text
    data = json.loads(raw)
    parsed = [CaptionOption(**item) for item in data]
    return parsed
