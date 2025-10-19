import json
from termcolor import cprint
from typing import List, Dict
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoTokenizer

from .base_client import BaseModelClient
from ..prompts import TableJSON


class VLLMOfflineClient(BaseModelClient):

    def __init__(self, model_name: str, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.9):
        cprint(f"Loading vLLM offline engine for: {model_name}", "yellow")
        cprint(f"  Tensor Parallel Size: {tensor_parallel_size}", "yellow")
        cprint(f"  GPU Memory Utilization: {gpu_memory_utilization}", "yellow")
        
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=10000, 
        )
        self.model_name = model_name
        

        json_schema = TableJSON.model_json_schema()
        self.sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=4096,
            stop=None,
            guided_decoding=GuidedDecodingParams(json=json_schema)
        )
        
        cprint(f"✓ vLLM Offline Client initialized for: {model_name}", "green")

    def generate_structured_json(self, prompt: str, max_retries: int = 3) -> TableJSON | None:
        results = self.generate_structured_json_batch([{"id": "single", "prompt": prompt}])
        return results.get("single")

    def generate_structured_json_batch(
        self, 
        prompts: List[Dict[str, str]], 
        max_retries: int = 2
    ) -> Dict[str, TableJSON | None]:
        results = {}
        
        for attempt in range(max_retries):
            
            tokenizer = self.llm.get_tokenizer()
            remaining_prompts_with_templated_text = []
            for p in prompts:
                if p["id"] not in results or results[p["id"]] is None:
                    messages_for_prompt = [{"role": "user", "content": p["prompt"]}]
                    templated_text = tokenizer.apply_chat_template(
                        messages_for_prompt,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False, 
                    )
                    remaining_prompts_with_templated_text.append({"id": p["id"], "prompt": templated_text})
            remaining_prompts = remaining_prompts_with_templated_text

            
            if not remaining_prompts:
                break
            
            cprint(f"\nBatch inference: {len(remaining_prompts)} prompts (attempt {attempt + 1})...", "cyan")
            
            try:
                prompt_texts = [p["prompt"] for p in remaining_prompts]
                prompt_ids = [p["id"] for p in remaining_prompts]

                outputs = self.llm.generate(prompt_texts, self.sampling_params)
   
                for i, output in enumerate(outputs):
                    prompt_id = prompt_ids[i]
                    generated_text = output.outputs[0].text
                    
                    try:
                        table_json = TableJSON.model_validate_json(generated_text)
                        results[prompt_id] = table_json
                        cprint(f"  ✓ {prompt_id}: Success", "green")
                    except Exception as e:
                        cprint(f"  ✗ {prompt_id}: Parse failed - {str(e)[:100]}", "yellow")
                        cprint(generated_text,"red")
                        results[prompt_id] = generated_text
                        
            except Exception as e:
                cprint(f"Batch inference failed: {e}", "red")
                for p in remaining_prompts:
                    if p["id"] not in results:
                        results[p["id"]] = None
        
        # Log final summary
        successful = sum(1 for r in results.values() if r is not None)
        cprint(f"\n✓ Batch complete: {successful}/{len(prompts)} successful", "green" if successful == len(prompts) else "yellow")
        
        failed = [pid for pid, result in results.items() if result is None]
        if failed:
            cprint(f"✗ Failed IDs: {', '.join(failed)}", "red")
            
        return results

    def __del__(self):
        if hasattr(self, 'llm'):
            del self.llm
            cprint("vLLM offline engine cleaned up", "yellow")