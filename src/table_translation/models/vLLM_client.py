import json
import time
from openai import OpenAI
from termcolor import cprint
from typing import Any, List, Dict

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

    def generate_structured_json_batch(
        self, 
        prompts: List[Dict[str, str]], 
        max_retries: int = 3
    ) -> Dict[str, TableJSON | None]:
        results = {}
        
        for attempt in range(max_retries):
            remaining_prompts = [p for p in prompts if p["id"] not in results or results[p["id"]] is None]
            
            if not remaining_prompts:
                break
                
            try:
                cprint(f"Batch processing {len(remaining_prompts)} prompts (attempt {attempt + 1})...", "cyan")
                
                batch_messages = [
                    {"role": "user", "content": p["prompt"]} 
                    for p in remaining_prompts
                ]
                
                responses = []
                for msg in batch_messages:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[msg],
                        max_tokens=4096,
                        temperature=0.1,
                        response_format={"type": "json_object"},
                    )
                    responses.append(response)
                
                # Parse results
                for i, response in enumerate(responses):
                    prompt_id = remaining_prompts[i]["id"]
                    try:
                        content = response.choices[0].message.content
                        table_json = TableJSON.model_validate_json(content)
                        results[prompt_id] = table_json
                        cprint(f"  ✓ {prompt_id}: Success", "green")
                    except Exception as e:
                        cprint(f"  ✗ {prompt_id}: Parse failed - {e}", "yellow")
                        results[prompt_id] = None
                        
            except Exception as e:
                cprint(f"Batch attempt {attempt + 1} failed: {e}", "yellow")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        failed = [pid for pid, result in results.items() if result is None]
        if failed:
            cprint(f"Failed to get valid JSON for: {', '.join(failed)}", "red")
            
        return results