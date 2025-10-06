import json
import time
from openai import OpenAI
from termcolor import cprint
from typing import Any

from .base_client import BaseModelClient
from ..prompts import TableJSON

class VLLMClient(BaseModelClient):
    def __init__(self, base_url: str, api_key: str, model_name: str):
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        cprint(f"VLLM Client initialized for model: {self.model_name}", "green")

    def generate_structured_json(self, prompt: str, max_retries: int = 3) -> TableJSON | None:
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4096,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                
                return TableJSON.model_validate_json(content)
            except Exception as e:
                cprint(f"vLLM attempt {attempt + 1} failed: {e}", "yellow")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    
        cprint(f"Failed to get valid JSON from vLLM after {max_retries} attempts.", "red")
        return None