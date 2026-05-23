"""
llm/client.py
LLM client — Gemini 2.0 Flash (google-genai SDK) primary, Groq llama3-70b-8192 fallback.
Supports streaming and JSON-mode generation.
"""

import json
import os
import re
import time

import google.genai as genai
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GEMINI_MODEL = "gemini-2.0-flash-lite"
GROQ_MODEL = "llama-3.3-70b-versatile"


class LLMClient:
    def __init__(self):
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        groq_key = os.getenv("GROQ_API_KEY", "")

        self._gemini_client = genai.Client(api_key=gemini_key) if gemini_key else None
        self.groq = Groq(api_key=groq_key) if groq_key else None

    # ------------------------------------------------------------------
    # Core generate
    # ------------------------------------------------------------------
    def generate(self, prompt: str, system: str = "", stream: bool = False):
        """
        Generate a response. If stream=True, yields text chunks.
        Falls back to Groq on any Gemini exception.
        """
        full_prompt = f"{system}\n\n{prompt}".strip() if system else prompt
        if self._gemini_client:
            try:
                if stream:
                    print("[llm] using gemini (stream)")
                    response = self._gemini_client.models.generate_content_stream(
                        model=GEMINI_MODEL, contents=full_prompt
                    )
                    for chunk in response:
                        if chunk.text:
                            yield chunk.text
                    return
                else:
                    print("[llm] using gemini")
                    response = self._gemini_client.models.generate_content(
                        model=GEMINI_MODEL, contents=full_prompt
                    )
                    return response.text
            except Exception as e:
                err_str = str(e)
                # If rate-limited, wait the retry delay (up to 15s) before falling back
                retry_match = re.search(r'retryDelay.*?(\d+)s', err_str)
                if retry_match:
                    wait = min(int(retry_match.group(1)), 15)
                    print(f"[llm] Gemini rate-limited, waiting {wait}s then trying Groq...")
                    time.sleep(wait)
                    # Retry Gemini once after waiting
                    try:
                        if stream:
                            response = self._gemini_client.models.generate_content_stream(
                                model=GEMINI_MODEL, contents=full_prompt
                            )
                            for chunk in response:
                                if chunk.text:
                                    yield chunk.text
                            return
                        else:
                            response = self._gemini_client.models.generate_content(
                                model=GEMINI_MODEL, contents=full_prompt
                            )
                            return response.text
                    except Exception as e2:
                        print(f"[llm] Gemini retry failed ({str(e2)[:80]}), using Groq")
                else:
                    print(f"[llm] Gemini failed ({str(e)[:120]}), falling back to Groq")

        # Groq fallback
        if not self.groq:
            raise RuntimeError("No LLM configured — set GEMINI_API_KEY or GROQ_API_KEY in .env")
        yield from self._groq_generate(prompt, system, stream)

    def _groq_generate(self, prompt: str, system: str, stream: bool):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if stream:
            print("[llm] using groq (stream)")
            resp = self.groq.chat.completions.create(
                model=GROQ_MODEL, messages=messages, stream=True
            )
            for chunk in resp:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        else:
            print("[llm] using groq")
            resp = self.groq.chat.completions.create(model=GROQ_MODEL, messages=messages)
            yield resp.choices[0].message.content

    # ------------------------------------------------------------------
    # JSON helper
    # ------------------------------------------------------------------
    def generate_json(self, prompt: str, system: str = "") -> dict:
        """Call generate (non-stream), strip markdown fences, parse JSON."""
        raw = self.generate(prompt, system=system, stream=False)
        # Consume generator if Groq fallback triggered
        if hasattr(raw, "__iter__") and not isinstance(raw, str):
            raw = "".join(raw)
        if not raw:
            return {}
        raw = raw.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[llm] JSON parse error: {e}  raw={raw[:200]}")
            return {}


# ------------------------------------------------------------------
# Smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    client = LLMClient()

    print("=== Non-stream test ===")
    result = client.generate("Say hello in exactly 5 words.")
    if hasattr(result, "__iter__") and not isinstance(result, str):
        result = "".join(result)
    print(f"Response: {result}")

    print("\n=== JSON test ===")
    data = client.generate_json(
        'Return a JSON object with keys "name" and "fact" about the Python programming language. '
        "Output ONLY the JSON, no markdown."
    )
    print(f"Parsed JSON: {data}")
