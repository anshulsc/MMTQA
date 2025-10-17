import json
import time
from typing import Dict, Any, Optional
from termcolor import cprint
import google.generativeai as genai

from src.configs import qa_config as qa_cfg
from .prompts import QA_GENERATION_PROMPT, GeneratedQACollection, EXAMPLE_TABLE, QA_EXAMPLE,ANSWER_FORMAT

class QAGenerator:
    # Class-level variable to track current API key index
    _current_key_index = 0
    # Class-level set to track processed table IDs
    _processed_tables = set()
    
    def __init__(self, table_id: str, table_data: Dict[str, Any]):
        self.table_id = table_id
        self.table_data = table_data
        self._configure_model()
    
    @classmethod
    def load_processed_tables(cls):
        """Load already processed table IDs from existing QA files"""
        cls._processed_tables.clear()
        for qa_file in qa_cfg.QA_PAIRS_DIR.glob("*_qa.json"):
            table_id = qa_file.stem.replace("_qa", "")
            cls._processed_tables.add(table_id)
        if cls._processed_tables:
            cprint(f"Loaded {len(cls._processed_tables)} already processed tables.", "green")
    
    @classmethod
    def is_processed(cls, table_id: str) -> bool:
        """Check if a table has already been processed"""
        return table_id in cls._processed_tables
    
    @classmethod
    def mark_as_processed(cls, table_id: str):
        """Mark a table as processed"""
        cls._processed_tables.add(table_id)

    def _configure_model(self):
        """Configure the model with the current API key"""
        current_key = qa_cfg.GEMINI_API_KEYS[QAGenerator._current_key_index]
        print(current_key)
        genai.configure(api_key=current_key)
        self.model = genai.GenerativeModel(
            qa_cfg.GEMINI_MODEL_NAME,
            generation_config={"response_mime_type": "application/json"}
        )

    def _rotate_api_key(self):
        """Rotate to the next API key in circular manner"""
        QAGenerator._current_key_index = (QAGenerator._current_key_index + 1) % len(qa_cfg.GEMINI_API_KEYS)
        cprint(f"  Rotating to API key #{QAGenerator._current_key_index + 1}", "cyan")
        self._configure_model()

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
                print(response)
                
                validated_data = GeneratedQACollection.model_validate_json(response.text)
                
                if len(validated_data.qa_pairs) == 0:
                    cprint(f"  [WARN] Gemini returned an empty list of QA pairs.", "yellow")
                    return None
                    
                cprint(f"  [SUCCESS] Successfully generated and validated {len(validated_data.qa_pairs)} QA pairs.", "green")
                return validated_data

            except Exception as e:
                error_str = str(e).lower()
                cprint(f"  [ERROR] Attempt {attempt + 1} failed: {e}", "red")
                
                if "quota" in error_str or "rate limit" in error_str or "resource exhausted" in error_str:
                    cprint("  Quota/Rate limit detected. Rotating API key...", "magenta")
                    self._rotate_api_key()
                    time.sleep(2)
                elif attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        cprint(f"  [FAIL] Failed to generate valid QA pairs for {self.table_id} after {max_retries} attempts.", "red")
        return None