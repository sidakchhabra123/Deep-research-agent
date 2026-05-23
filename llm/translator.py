"""
llm/translator.py
Sarvam translation client for the Streamlit translation sandwich.
"""

from __future__ import annotations

import os
import re
import unicodedata


_BAD_GLYPHS = {"\ufffd", "□", "☐", "☒", "�"}


def _clean_text(text: str) -> str:
    cleaned = []
    for ch in text:
        if ch in "\n\t":
            cleaned.append(ch)
            continue
        if unicodedata.category(ch).startswith("C"):
            cleaned.append(" ")
            continue
        cleaned.append(ch)
    return " ".join("".join(cleaned).split())


def _looks_corrupt(text: str) -> bool:
    if not text:
        return True
    bad_count = sum(text.count(ch) for ch in _BAD_GLYPHS)
    if bad_count >= 10:
        return True
    return bad_count > 0 and bad_count / max(len(text), 1) > 0.05


def _protect_citations(text: str) -> tuple[str, dict[str, str]]:
    citations = {}

    def repl(match):
        token = f"__CITATION_{len(citations)}__"
        citations[token] = match.group(0)
        return token

    return re.sub(r"\[\d+\]", repl, text), citations


def _restore_citations(text: str, citations: dict[str, str]) -> str:
    restored = text
    for token, citation in citations.items():
        restored = restored.replace(token, citation)
        restored = restored.replace(token.replace("_", " "), citation)
    return restored


class SarvamTranslator:
    def __init__(self):
        self.api_key = os.environ.get("SARVAM_API_KEY", "")
        self.endpoint = "https://api.sarvam.ai/translate"

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if not text or source_lang == target_lang:
            return text
        original_text = text
        text = _clean_text(text)
        text, citations = _protect_citations(text)
        if not self.api_key:
            print("[translator] SARVAM_API_KEY is not set; returning original text")
            return original_text

        try:
            import requests

            response = requests.post(
                self.endpoint,
                headers={
                    "Content-Type": "application/json",
                    "api-subscription-key": self.api_key,
                },
                json={
                    "input": text,
                    "source_language_code": source_lang,
                    "target_language_code": target_lang,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            translated = (
                data.get("translated_text")
                or data.get("output")
                or data.get("translation")
                or data.get("text")
            )
            if isinstance(translated, list):
                translated = " ".join(str(item) for item in translated if item)
            if not translated and isinstance(data.get("translations"), list):
                translated = " ".join(
                    str(item.get("translated_text") or item.get("text") or "")
                    for item in data["translations"]
                    if isinstance(item, dict)
                ).strip()
            if isinstance(translated, str) and translated.strip():
                translated = _clean_text(translated)
                if _looks_corrupt(translated):
                    print("[translator] Sarvam returned unreadable glyphs; returning original text")
                    return original_text
                return _restore_citations(translated, citations)
            print(f"[translator] Unexpected Sarvam response shape: {data}")
        except Exception as e:
            print(f"[translator] Translation failed: {e}")

        return original_text
