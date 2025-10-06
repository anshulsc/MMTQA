import json
import time
import google.generativeai as genai
from termcolor import cprint
from typing import Any

from .base_client import BaseModelClient
from ..prompts import TableJSON

class GeminiClient(BaseModelClient):
    def __init__(self, api_key: str, model_name: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name,
            generation_config={"response_mime_type": "application/json"}
        )
        self.model_name = model_name
        cprint(f"Gemini Client initialized for model: {self.model_name}", "green")

    def generate_structured_json(self, prompt: str, max_retries: int = 3) -> TableJSON | None:
        """Generates structured JSON using Gemini's JSON mode."""
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                # Gemini's JSON mode returns a clean string
                content = response.text
                return TableJSON.model_validate_json(content)
            except Exception as e:
                cprint(f"Gemini attempt {attempt + 1} failed: {e}", "yellow")
                if "rate limit" in str(e).lower():
                    time.sleep(15) # Longer wait for rate limits
                else:
                    time.sleep(2 ** attempt)
        cprint(f"Failed to get valid JSON from Gemini after {max_retries} attempts.", "red")
        return None