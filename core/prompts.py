from __future__ import annotations

def build_caption_prompt(n:int,
                         styles:list[str],
                         platform:str|None,
                         include_hashtags:bool,
                         max_chars:int|None,
                         extra_keywords:str|None) -> str:
    styles_str = ", ".join(styles) if styles else "diverse"
    max_char_rule = f"Limit each caption to <= {max_chars} characters." if max_chars else ""
    platform_hint = f"Optimize tone for {platform}." if platform else ""
    hashtag_rule = (
        "Include 3–8 relevant, non-spammy hashtags per caption in a 'hashtags' array."
        if include_hashtags else
        "Return an empty 'hashtags' array."
    )
    keyword_hint = f"Try to incorporate these keywords when relevant: {extra_keywords}." if extra_keywords else ""

    return f"""
You write concise, human-quality captions from an image.

Return EXACTLY {n} caption options as JSON per the schema (no extra text).
Goals:
- Make captions natural, specific to the image, and not generic.
- Avoid hallucinating details that aren't visible.
- Add a tiny dose of personality; no cringe.
- Provide 'alt_text' variant suitable for accessibility (describe what's actually visible).
- Styles to vary across options: {styles_str}.
- {platform_hint}
- {max_char_rule}
- {keyword_hint}
- {hashtag_rule}

Follow the schema strictly.
"""
