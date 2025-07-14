import os
import itertools
from typing import List, Dict, Any

try:
    import openai
except ImportError:
    openai = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


def _convert_messages_to_gemini(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Convert OpenAI-style messages to Gemini format."""
    return [{"role": m["role"], "parts": [m["content"]]} for m in messages]


class LLMBackend:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "openai").lower()
        gemini_keys = os.getenv("GEMINI_API_KEYS", "").split(',')
        self._gemini_cycle = itertools.cycle([k for k in gemini_keys if k])

    def _ask_openai(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        if openai is None:
            raise ImportError("openai package not installed")
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=kwargs.get("temperature", 0.5),
            max_tokens=kwargs.get("max_tokens", 1024),
        )
        return response

    def _ask_gemini(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        if genai is None:
            raise ImportError("google-generativeai package not installed")
        api_key = next(self._gemini_cycle, None)
        if api_key:
            genai.configure(api_key=api_key)
        history = _convert_messages_to_gemini(messages)
        response = genai.chat.Completions.create(
            model="gemini-pro",
            contents=history,
            temperature=kwargs.get("temperature", 0.5),
            max_output_tokens=kwargs.get("max_tokens", 1024),
        )
        return {"choices": [{"message": {"content": response.text}}]}

    def ask_llm(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        if self.provider == "gemini":
            return self._ask_gemini(messages, **kwargs)
        return self._ask_openai(messages, **kwargs)
