import json
import time
from typing import Dict, Any, Optional
from termcolor import cprint
import google.generativeai as genai

from src.configs import qa_config as qa_cfg
from .prompts import QA_GENERATION_PROMPT, GeneratedQACollection, EXAMPLE_TABLE, QA_EXAMPLE,ANSWER_FORMAT

class QAGenerator:
    def __init__(self, table_id: str, table_data: Dict[str, Any]):
        self.table_id = table_id
        self.table_data = table_data
        
        genai.configure(api_key=qa_cfg.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(
            qa_cfg.GEMINI_MODEL_NAME,
            generation_config={"response_mime_type": "application/json"}
        )


    def generate(self, max_retries: int = 3) -> Optional[GeneratedQACollection]:
        table_json_string = json.dumps(self.table_data, indent=2)
        
        prompt = QA_GENERATION_PROMPT.format(
            num_questions=qa_cfg.NUM_QUESTIONS_PER_TABLE,
            answer_format=ANSWER_FORMAT,
            qa_example=QA_EXAMPLE,
            example_table=EXAMPLE_TABLE,
            table_as_json_string=table_json_string  
        )

        for attempt in range(max_retries):
            try:
                cprint(f"  Attempt {attempt + 1}/{max_retries} to generate QAs for {self.table_id}...", "yellow")
                response = self.model.generate_content(prompt)
                
                validated_data = GeneratedQACollection.model_validate_json(response.text)
                
                if len(validated_data.qa_pairs) == 0:
                    cprint(f"  [WARN] Gemini returned an empty list of QA pairs.", "yellow")
                    return None
                    
                cprint(f"  [SUCCESS] Successfully generated and validated {len(validated_data.qa_pairs)} QA pairs.", "green")
                return validated_data

            except Exception as e:
                cprint(f"  [ERROR] Attempt {attempt + 1} failed: {e}", "red")
                if "rate limit" in str(e).lower():
                     cprint("  Rate limit reached. Waiting for 60 seconds...", "magenta")
                     time.sleep(60)
                elif attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        cprint(f"  [FAIL] Failed to generate valid QA pairs for {self.table_id} after {max_retries} attempts.", "red")
        return None