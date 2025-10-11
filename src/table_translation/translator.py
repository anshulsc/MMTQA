import json
from pathlib import Path
from termcolor import cprint
from typing import Dict

from src.configs import translation_config as cfg
from src.utils.table_metrics import calculate_bleu
from .models.vLLM_client import VLLMClient
from .models.gemini_client import GeminiClient
from .prompts import INITIAL_TRANSLATION_PROMPT, REFINEMENT_PROMPT, TableJSON

class TableTranslator:
    def __init__(self, table_id: str, table_path: Path, vllm_client: VLLMClient, gemini_client: GeminiClient):
        self.table_id = table_id
        self.table_path = table_path
        self.vllm_client = vllm_client
        self.gemini_client = gemini_client
        self.original_table_data = json.loads(table_path.read_text())

    def _save_checkpoint(self, step_name: str, lang_code: str, data: dict):
        checkpoint_dir = cfg.CHECKPOINTS_DIR / self.table_id
        checkpoint_dir.mkdir(exist_ok=True)
        path = checkpoint_dir / f"{step_name}_{lang_code}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def run_batch_translation(self, languages: Dict[str, str]):
        """
        Run translation for all languages in batch mode.
        
        Args:
            languages: Dict mapping lang_code to lang_name (e.g., {"hi": "Hindi", "es": "Spanish"})
        """
        cprint(f"\n{'='*60}", "cyan")
        cprint(f"Processing table: {self.table_id} ({len(languages)} languages)", "cyan")
        cprint(f"{'='*60}", "cyan")
        
        # Save original
        self._save_checkpoint("original", "en", self.original_table_data)
        cprint("\n[STEP 1] Batch Initial Translation...", "blue", attrs=["bold"])
        
        initial_prompts = []
        for lang_code, lang_name in languages.items():
            prompt = INITIAL_TRANSLATION_PROMPT.format(
                source_language=cfg.SOURCE_LANG_NAME,
                target_language=lang_name,
                table_json_string=json.dumps(self.original_table_data, indent=2)
            )
            initial_prompts.append({"id": lang_code, "prompt": prompt})
        
        initial_results = self.vllm_client.generate_structured_json_batch(initial_prompts)
        
        # Save step 1 checkpoints
        for lang_code, result in initial_results.items():
            if result:
                
                self._save_checkpoint("step1_initial_translation", lang_code, result.model_dump())
        
        successful_langs = {lc: ln for lc, ln in languages.items() if initial_results.get(lc)}
        failed_langs = {lc: ln for lc, ln in languages.items() if not initial_results.get(lc)}
        
        cprint(f"✓ Step 1 complete: {len(successful_langs)}/{len(languages)} successful", "green")
        if failed_langs:
            cprint(f"✗ Failed languages: {', '.join(failed_langs.keys())}", "red")
        
        cprint("\n[STEP 2] Sequential Refinement (Gemini)...", "blue", attrs=["bold"])
        
        refined_results = {}
        
        import time 
        gemini_call_count = 0
        last_rate_limit_reset_time = time.time()
        RATE_LIMIT_PER_MINUTE = 10
        RATE_LIMIT_PERIOD_SECONDS = 60
        
        for lang_code in successful_langs.keys():
            lang_name = languages[lang_code]
            cprint(f"  Refining {lang_name} ({lang_code})...", "cyan")
            

            current_time = time.time()
            
            if current_time - last_rate_limit_reset_time >= RATE_LIMIT_PERIOD_SECONDS:
                gemini_call_count = 0
                last_rate_limit_reset_time = current_time
            
            # If the call limit for the current minute has been reached, wait
            if gemini_call_count >= RATE_LIMIT_PER_MINUTE:
                time_to_wait = RATE_LIMIT_PERIOD_SECONDS - (current_time - last_rate_limit_reset_time)
                if time_to_wait > 0:
                    cprint(f"    Gemini rate limit ({RATE_LIMIT_PER_MINUTE} calls/{RATE_LIMIT_PERIOD_SECONDS}s) hit. Waiting for {time_to_wait:.2f}s...", "yellow")
                    time.sleep(time_to_wait)
                    
                    gemini_call_count = 0
                    last_rate_limit_reset_time = time.time() 
            
            prompt = REFINEMENT_PROMPT.format(
                original_table_json=json.dumps(self.original_table_data, indent=2),
                translated_table_json=json.dumps(initial_results[lang_code].model_dump(), indent=2),
                target_language=lang_name
            )
            
            refined = self.gemini_client.generate_structured_json(prompt)
            gemini_call_count += 1 
            
            if refined:
                refined_results[lang_code] = refined
                self._save_checkpoint("step2_refined_translation", lang_code, refined.model_dump())
                cprint(f"    ✓ {lang_code} refined", "green")
            else:
                cprint(f"    ✗ {lang_code} refinement failed", "red")
        
        cprint(f"✓ Step 2 complete: {len(refined_results)}/{len(successful_langs)} refined", "green")
        
        # ========================================
        # STEP 3: BATCH BACK-TRANSLATION
        # ========================================
        cprint("\n[STEP 3] Batch Back-Translation...", "blue", attrs=["bold"])
        
        back_prompts = []
        for lang_code, refined_table in refined_results.items():
            lang_name = languages[lang_code]
            prompt = INITIAL_TRANSLATION_PROMPT.format(
                source_language=lang_name,
                target_language=cfg.SOURCE_LANG_NAME,
                table_json_string=json.dumps(refined_table.model_dump(), indent=2)
            )
            back_prompts.append({"id": lang_code, "prompt": prompt})
        
        back_results = self.vllm_client.generate_structured_json_batch(back_prompts)
        
        # Save step 3 checkpoints
        for lang_code, result in back_results.items():
            if result:
                self._save_checkpoint("step3_back_translation", lang_code, result.model_dump())
        
        cprint(f"✓ Step 3 complete: {len([r for r in back_results.values() if r])}/{len(refined_results)} back-translated", "green")
        
        # ========================================
        # STEP 4: BLEU EVALUATION & SAVE
        # ========================================
        cprint("\n[STEP 4] BLEU Evaluation & Saving...", "blue", attrs=["bold"])
        
        kept_count = 0
        dropped_count = 0
        
        for lang_code in refined_results.keys():
            lang_name = languages[lang_code]
            
            if lang_code not in back_results or not back_results[lang_code]:
                cprint(f"  ✗ {lang_code}: No back-translation available", "red")
                continue
            
            bleu_score = calculate_bleu(
                self.original_table_data, 
                back_results[lang_code].model_dump()
            )
            decision = "KEEP" if bleu_score >= cfg.BLEU_THRESHOLD else "DROP"
            
            metadata = {
                "table_id": self.table_id,
                "lang_code": lang_code,
                "bleu_score": bleu_score,
                "threshold": cfg.BLEU_THRESHOLD,
                "decision": decision
            }
            
            # Save metadata
            meta_dir = cfg.TRANSLATION_METADATA_DIR / self.table_id
            meta_dir.mkdir(exist_ok=True)
            with open(meta_dir / f"{lang_code}.json", 'w') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
            
            if decision == "KEEP":
                final_dir = cfg.TRANSLATED_TABLES_DIR / self.table_id
                final_dir.mkdir(exist_ok=True)
                with open(final_dir / f"{lang_code}.json", 'w') as f:
                    json.dump(refined_results[lang_code].model_dump(), f, indent=2, ensure_ascii=False)
                cprint(f"  ✓ {lang_code}: BLEU={bleu_score:.4f} (KEPT)", "green")
                kept_count += 1
            else:
                cprint(f"  ✗ {lang_code}: BLEU={bleu_score:.4f} (DROPPED)", "yellow")
                dropped_count += 1
        
        cprint(f"\n{'='*60}", "cyan")
        cprint(f"Table {self.table_id} complete: {kept_count} kept, {dropped_count} dropped", "cyan")
        cprint(f"{'='*60}\n", "cyan")