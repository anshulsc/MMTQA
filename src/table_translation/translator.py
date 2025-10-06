import json
from pathlib import Path
from termcolor import cprint

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

    def run_for_language(self, lang_code: str, lang_name: str):
        cprint(f"  Translating to {lang_name} ({lang_code})...", "cyan")
        
        self._save_checkpoint("original","en",  self.original_table_data )

        prompt1 = INITIAL_TRANSLATION_PROMPT.format(
            source_language=cfg.SOURCE_LANG_NAME,
            target_language=lang_name,
            table_json_string=json.dumps(self.original_table_data, indent=2)
        )
        
        initial_translated = self.vllm_client.generate_structured_json(prompt1)
        print(initial_translated)
        
        if not initial_translated:
            cprint(f"    [FAIL] Step 1: Initial translation failed for {lang_name}.", "red")
            return
        self._save_checkpoint("step1_initial_translation", lang_code, initial_translated.model_dump())
        cprint("    [OK] Step 1: Initial translation complete.", "green")


        prompt2 = REFINEMENT_PROMPT.format(
            original_table_json=json.dumps(self.original_table_data, indent=2),
            translated_table_json=json.dumps(initial_translated.model_dump(), indent=2),
            target_language=lang_name
        )
        refined_translated = self.gemini_client.generate_structured_json(prompt2)
        if not refined_translated:
            cprint(f"    [FAIL] Step 2: Refinement failed for {lang_name}.", "red")
            return
        self._save_checkpoint("step2_refined_translation", lang_code, refined_translated.model_dump())
        cprint("    [OK] Step 2: Refinement complete.", "green")


        prompt3 = INITIAL_TRANSLATION_PROMPT.format(
            source_language=lang_name,
            target_language=cfg.SOURCE_LANG_NAME,
            table_json_string=json.dumps(refined_translated.model_dump(), indent=2)
        )
        back_translated = self.vllm_client.generate_structured_json(prompt3)
        if not back_translated:
            cprint(f"    [FAIL] Step 3: Back-translation failed for {lang_name}.", "red")
            return
        self._save_checkpoint("step3_back_translation", lang_code, back_translated.model_dump())
        cprint("    [OK] Step 3: Back-translation complete.", "green")


        bleu_score = calculate_bleu(self.original_table_data, back_translated.model_dump())
        decision = "KEEP" if bleu_score >= cfg.BLEU_THRESHOLD else "DROP"
        
        metadata = {
            "table_id": self.table_id,
            "lang_code": lang_code,
            "bleu_score": bleu_score,
            "threshold": cfg.BLEU_THRESHOLD,
            "decision": decision
        }
        
        
        meta_dir = cfg.TRANSLATION_METADATA_DIR / self.table_id
        meta_dir.mkdir(exist_ok=True)
        with open(meta_dir / f"{lang_code}.json", 'w') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        if decision == "KEEP":
            final_dir = cfg.TRANSLATED_TABLES_DIR / self.table_id
            final_dir.mkdir(exist_ok=True)
            with open(final_dir / f"{lang_code}.json", 'w') as f:
                json.dump(refined_translated.model_dump(), f, indent=2, ensure_ascii=False)
            cprint(f"    [OK] Step 4: BLEU score is {bleu_score:.4f} (>= {cfg.BLEU_THRESHOLD}). Table kept.", "green")
        else:
            cprint(f"    [DROP] Step 4: BLEU score is {bleu_score:.4f} (< {cfg.BLEU_THRESHOLD}). Table dropped.", "yellow")