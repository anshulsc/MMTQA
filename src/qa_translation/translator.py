import json
import time
from typing import Dict, Any, Optional, Set
from collections import deque
from datetime import datetime, timedelta
from termcolor import cprint
import google.generativeai as genai
from pathlib import Path

from src.configs import qa_translation_config as cfg
from .prompts import QA_TRANSLATION_PROMPT, TranslatedQA


class APIKeyRotator:
    """Manages API keys - uses one key until quota exceeded, then shifts to next"""
    
    def __init__(self, api_keys: list, requests_per_minute: int = 12):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.requests_per_minute = requests_per_minute
        
        # Track request count for current key (resets after wait)
        self.current_key_request_count = 0
        
        # Track keys that have exceeded quota (blacklist them)
        self.quota_exceeded_keys: Set[int] = set()
        
    def get_current_key(self) -> str:
        """Get the current API key"""
        return self.api_keys[self.current_key_index]
    
    def get_current_key_index(self) -> int:
        """Get the current key index for logging"""
        return self.current_key_index
    
    def mark_quota_exceeded(self, key_index: int):
        """Mark a key as having exceeded quota and rotate to next available key"""
        self.quota_exceeded_keys.add(key_index)
        cprint(f"    [QUOTA] Key #{key_index + 1} marked as quota exceeded. {len(self.quota_exceeded_keys)}/{len(self.api_keys)} keys exhausted.", "red")
        
        # Rotate to next available key
        if self.has_available_keys():
            self._rotate_to_next_available_key()
    
    def has_available_keys(self) -> bool:
        """Check if there are any keys left that haven't exceeded quota"""
        return len(self.quota_exceeded_keys) < len(self.api_keys)
    
    def _rotate_to_next_available_key(self):
        """Rotate to the next key that hasn't exceeded quota"""
        attempts = 0
        while attempts < len(self.api_keys):
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            attempts += 1
            
            if self.current_key_index not in self.quota_exceeded_keys:
                self.current_key_request_count = 0  # Reset counter for new key
                cprint(f"    [ROTATION] Switched to API key #{self.current_key_index + 1}", "cyan")
                return
        
        raise Exception("All API keys have exceeded their quota.")
    
    def check_and_wait_if_needed(self):
        """Check if we need to wait after 10 requests"""
        if self.current_key_request_count >= self.requests_per_minute:
            cprint(f"    [RATE LIMIT] Reached {self.requests_per_minute} requests on key #{self.current_key_index + 1}. Waiting 60 seconds...", "yellow")
            time.sleep(30)
            self.current_key_request_count = 0  # Reset counter after wait
            cprint(f"    [RESUMED] Continuing with key #{self.current_key_index + 1}", "green")
    
    def record_request(self):
        """Record that a request was made with the current key"""
        self.current_key_request_count += 1


class TranslationTracker:
    """Tracks completed translations to avoid repeating work"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.completed_cache: Dict[str, Set[str]] = {}
    
    def _get_cache_key(self, lang_code: str) -> str:
        """Get cache key for language code"""
        return lang_code
    
    def is_completed(self, lang_code: str, table_id: str) -> bool:
        """Check if translation is already completed"""
        cache_key = self._get_cache_key(lang_code)
        
        # Load cache if not already loaded
        if cache_key not in self.completed_cache:
            self.completed_cache[cache_key] = self._load_completed_files(lang_code)
        
        return table_id in self.completed_cache[cache_key]
    
    def _load_completed_files(self, lang_code: str) -> Set[str]:
        """Load list of completed table IDs for a language"""
        lang_dir = self.output_dir / lang_code
        if not lang_dir.exists():
            return set()
        
        completed = set()
        for json_file in lang_dir.glob("*.json"):
            # Extract table_id from filename (remove _qa.json suffix)
            table_id = json_file.stem.replace("_qa", "")
            completed.add(table_id)
        
        return completed
    
    def mark_completed(self, lang_code: str, table_id: str):
        """Mark a translation as completed"""
        cache_key = self._get_cache_key(lang_code)
        if cache_key not in self.completed_cache:
            self.completed_cache[cache_key] = set()
        self.completed_cache[cache_key].add(table_id)


class QATranslator:
    """Translates QA pairs from English to target languages with API key rotation"""
    
    # Class-level shared resources
    _key_rotator = None
    _translation_tracker = None
    
    @classmethod
    def _get_key_rotator(cls):
        """Get or create the shared key rotator"""
        if cls._key_rotator is None:
            requests_per_min = getattr(cfg, 'REQUESTS_PER_MINUTE', 10)
            cls._key_rotator = APIKeyRotator(
                api_keys=cfg.GEMINI_API_KEYS,
                requests_per_minute=requests_per_min
            )
        return cls._key_rotator
    
    @classmethod
    def _get_translation_tracker(cls):
        """Get or create the shared translation tracker"""
        if cls._translation_tracker is None:
            cls._translation_tracker = TranslationTracker(cfg.OUTPUT_DIR)
        return cls._translation_tracker
    
    def __init__(self, english_qa_pair: Dict[str, Any], context_table: Dict[str, Any]):
        self.english_qa_pair = english_qa_pair
        self.context_table = context_table
        self.key_rotator = self._get_key_rotator()
        self.tracker = self._get_translation_tracker()
        self.model = None
        self._configure_model()
    
    def _configure_model(self):
        """Configure Gemini model with current API key"""
        current_key = self.key_rotator.get_current_key()
        genai.configure(api_key=current_key)
        self.model = genai.GenerativeModel(
            cfg.GEMINI_MODEL_NAME,
            generation_config={"response_mime_type": "application/json"}
        )

    def translate(self, target_language: str, max_retries: int = 3) -> Optional[TranslatedQA]:
        """
        Translate the QA pair to the target language with automatic key rotation
        
        Args:
            target_language: Name of target language (e.g., "Spanish", "French")
            max_retries: Maximum number of retry attempts per error
            
        Returns:
            TranslatedQA object or None if translation fails
        """
        # Prepare the context table and QA pair as JSON strings
        context_table_json = json.dumps(self.context_table, indent=2, ensure_ascii=False)
        english_qa_json = json.dumps({
            "question": self.english_qa_pair.get("question", ""),
            "answer": self.english_qa_pair.get("answer", []),
            "question_type": self.english_qa_pair.get("question_type", "value")
        }, indent=2, ensure_ascii=False)
        
        # Format the prompt
        prompt = QA_TRANSLATION_PROMPT.format(
            target_language=target_language,
            context_table_json=context_table_json,
            english_qa_json=english_qa_json
        )
        
        total_attempts = 0
        max_total_attempts = max_retries * len(cfg.GEMINI_API_KEYS)
        
        while total_attempts < max_total_attempts:
            try:
                # Check if all keys are exhausted before attempting
                if not self.key_rotator.has_available_keys():
                    cprint(f"    [CRITICAL] All {len(cfg.GEMINI_API_KEYS)} API keys have exceeded quota!", "red")
                    return None
                
                # Check and wait if we've reached request limit for current key
                self.key_rotator.check_and_wait_if_needed()
                
                key_index = self.key_rotator.get_current_key_index()
                
                # Generate translation
                response = self.model.generate_content(prompt)
                
                # Record successful request
                self.key_rotator.record_request()
                
                # Validate the response using Pydantic model
                validated_translation = TranslatedQA.model_validate_json(response.text)
                
                cprint(f"    [SUCCESS] Translation to {target_language} completed (Key #{key_index + 1}, Request #{self.key_rotator.current_key_request_count}).", "green")
                return validated_translation

            except Exception as e:
                error_msg = str(e).lower()
                total_attempts += 1
                key_index = self.key_rotator.get_current_key_index()
                
                cprint(f"    [ERROR] Attempt {total_attempts} (Key #{key_index + 1}): {e}", "red")
                
                # Handle quota exceeded - mark key and rotate to next
                if any(keyword in error_msg for keyword in ["quota exceeded", "resource exhausted", "429"]):
                    self.key_rotator.mark_quota_exceeded(key_index)
                    
                    # Check if all keys are exhausted
                    if not self.key_rotator.has_available_keys():
                        cprint(f"    [CRITICAL] All {len(cfg.GEMINI_API_KEYS)} API keys have exceeded quota!", "red")
                        return None
                    
                    # Reconfigure model with new key
                    self._configure_model()
                    
                    # Add small delay after quota error
                    delay = getattr(cfg, 'QUOTA_ERROR_DELAY', 2)
                    time.sleep(delay)
                
                # Handle rate limit errors (shouldn't happen with our logic, but just in case)
                elif "rate limit" in error_msg:
                    cprint("    Rate limit detected, waiting 60 seconds...", "yellow")
                    time.sleep(60)
                    self.key_rotator.current_key_request_count = 0
                
                # Handle validation or other errors with exponential backoff
                elif total_attempts < max_total_attempts:
                    wait_time = min(2 ** (total_attempts % 5), 16)  # Cap at 16 seconds
                    cprint(f"    Waiting {wait_time}s before retry...", "yellow")
                    time.sleep(wait_time)
        
        cprint(f"    [FAIL] Failed to translate to {target_language} after {total_attempts} attempts.", "red")
        return None
    
    @classmethod
    def should_skip_translation(cls, lang_code: str, table_id: str) -> bool:
        """Check if translation should be skipped (already completed)"""
        tracker = cls._get_translation_tracker()
        return tracker.is_completed(lang_code, table_id)
    
    @classmethod
    def mark_translation_complete(cls, lang_code: str, table_id: str):
        """Mark translation as complete"""
        tracker = cls._get_translation_tracker()
        tracker.mark_completed(lang_code, table_id)


class QAGenerator:
    """Generates QA pairs from tables (placeholder for your existing code)"""
    
    def __init__(self, table_id: str, table_data: Dict[str, Any]):
        self.table_id = table_id
        self.table_data = table_data

    def generate(self, max_retries: int = 3):
        """Generate QA pairs from table data"""
        pass