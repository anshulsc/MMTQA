import time
from typing import Optional
from termcolor import cprint

class GeminiClient:
    def __init__(self, api_keys: list[str], model_name: str):
        if not api_keys:
            raise ValueError("At least one API key must be provided")
        
        self.api_keys = api_keys
        self.model_name = model_name
        self.current_key_index = 0
        self.failed_keys = set() 
        self._init_client()
    
    def _init_client(self):
        import google.generativeai as genai
        
        current_key = self.api_keys[self.current_key_index]
        genai.configure(api_key=current_key)
        self.client = genai.GenerativeModel(self.model_name)
        cprint(f"Using API key #{self.current_key_index + 1}/{len(self.api_keys)}", "cyan")
    
    def _rotate_key(self):
        self.failed_keys.add(self.current_key_index)
        
        if len(self.failed_keys) >= len(self.api_keys):
            cprint("All API keys have reached quota!", "red", attrs=["bold"])
            return False
        

        attempts = 0
        while attempts < len(self.api_keys):
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            
            if self.current_key_index not in self.failed_keys:
                cprint(f"Rotating to API key #{self.current_key_index + 1}/{len(self.api_keys)}", "yellow")
                self._init_client()
                return True
            
            attempts += 1
        
        return False
    
    def generate_structured_json(self, prompt: str, max_retries: int = None):

        if max_retries is None:
            max_retries = len(self.api_keys)
        
        from ..prompts import TableJSON
        
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = self.client.generate_content(
                    prompt,
                    generation_config={
                        "response_mime_type": "application/json",
                        "temperature": 0.2
                    }
                )
                
                result = TableJSON.model_validate_json(response.text)
                
                if self.current_key_index in self.failed_keys:
                    self.failed_keys.discard(self.current_key_index)
                
                return result
                
            except Exception as e:
                error_msg = str(e).lower()
                
                if any(keyword in error_msg for keyword in ["quota", "rate limit", "resource exhausted", "429"]):
                    cprint(f"API key #{self.current_key_index + 1} quota reached: {e}", "yellow")
                    

                    if self._rotate_key():
                        retry_count += 1
                        time.sleep(1) 
                        continue
                    else:
                        cprint("All API keys exhausted. Cannot continue.", "red")
                        return None
                
                else:
                    cprint(f"Error generating content: {e}", "red")
                    return None
        
        cprint(f"Max retries ({max_retries}) reached. All keys may be exhausted.", "red")
        return None
    
    def reset_failed_keys(self):
        self.failed_keys.clear()
        cprint("Reset failed API keys tracking", "green")