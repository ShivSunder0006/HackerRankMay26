"""
llm_client.py — LLM abstraction: Groq primary + Gemini 1.5 Flash fallback.

Groq: full OpenAI-compatible chat completions with tool calling.
Gemini: simple direct-prompt fallback (no tool calling needed — caller
        pre-builds the prompt with retrieved context already included).
"""

import os
import time
from typing import Any, List, Dict, Optional

import google.generativeai as genai
from openai import OpenAI, RateLimitError, APIError, APIConnectionError


class LLMClient:
    def __init__(self):
        featherless_key = os.getenv("FEATHERLESS_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")

        if not featherless_key:
            raise ValueError("FEATHERLESS_API_KEY not set. Check your .env file.")

        self.primary = OpenAI(
            base_url="https://api.featherless.ai/v1",
            api_key=featherless_key
        )
        self.primary_model = "Qwen/Qwen3-0.6B"

        if gemini_key:
            genai.configure(api_key=gemini_key)
            self.gemini = genai.GenerativeModel("gemini-flash-latest")
            self._gemini_enabled = True
        else:
            self.gemini = None
            self._gemini_enabled = False
            print("Warning: GEMINI_API_KEY not set — Gemini fallback disabled.")

    # ── Primary (Featherless) ─────────────────────────────────────────────────────────────

    def primary_chat(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.1,
    ) -> Any:
        """
        Single primary chat call with 1 automatic retry on transient errors.
        Raises on second failure so the agent can fall back to Gemini.
        """
        for attempt in range(2):
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.primary_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 2048,
                    "seed": 42,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"

                return self.primary.chat.completions.create(**kwargs)

            except (RateLimitError, APIConnectionError) as e:
                if attempt == 0:
                    print(f"  [Featherless] {type(e).__name__} — retrying in 3s...")
                    time.sleep(3)
                    continue
                raise  # propagate to caller → triggers Gemini fallback

            except APIError as e:
                if attempt == 0 and getattr(e, "status_code", 0) >= 500:
                    print(f"  [Featherless] Server error {e.status_code} — retrying in 3s...")
                    time.sleep(3)
                    continue
                raise


    # ── Gemini ───────────────────────────────────────────────────────────────

    def gemini_direct(self, prompt: str) -> str:
        """
        Simple Gemini fallback: single prompt → text response.
        The caller is responsible for embedding retrieved context in the prompt.
        """
        if not self._gemini_enabled:
            raise RuntimeError("Gemini is not configured (GEMINI_API_KEY missing).")
        try:
            response = self.gemini.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=2048,
                ),
            )
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini call failed: {e}") from e
