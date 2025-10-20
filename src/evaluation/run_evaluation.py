from datetime import datetime
from pathlib import Path
from termcolor import cprint


from .config import get_config
from src.evaluation.models.qwen import QwenModel
from src.evaluation.data_loader import load_benchmark_data

MODEL_EVALUATOR_MAPPING = {
    "qwen": QwenModel,
}

def get_output_filepath(cfg) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_safe = cfg.model.model_path.replace("/", "_")
    lang_str = cfg.dataset.lang_code
    
   
    output_file_base = cfg.output_file_template.format(
        model_name=model_name_safe,
        dataset_name=cfg.dataset.name,
        image_type=cfg.dataset.image_type
    )

    return f"{output_file_base}_{lang_str}_{timestamp}.jsonl"

def main():
    cfg = get_config()

    cprint("--- MultiTableQA Evaluation Pipeline (Python Config) ---", "magenta", attrs=["bold"])
    cprint("Running with the following configuration:", "yellow")
    cprint(f"  - Model: {cfg.model.name} ({cfg.model.model_path})", "yellow")
    cprint(f"  - Dataset: {cfg.dataset.data_file}", "yellow")
    cprint(f"  - Images: {cfg.dataset.image_type} (lang: {cfg.dataset.lang_code})", "yellow")
    cprint("-" * 50, "magenta")

    model_name = cfg.model.name
    model_class = MODEL_EVALUATOR_MAPPING.get(model_name)
    if model_class is None:
        raise ValueError(f"Unsupported model: {model_name}. Supported: {list(MODEL_EVALUATOR_MAPPING.keys())}")

    try:
        # 1. Load Data
        data = load_benchmark_data(
            data_file_path=cfg.dataset.data_file,
            image_type=cfg.dataset.image_type,
            lang_code_filter=cfg.dataset.lang_code
        )
        if not data:
            cprint("No data loaded for the specified criteria. Exiting.", "red")
            return

        # 2. Initialize Model (this loads the VLLM engine into GPU memory)
        model = model_class(cfg)

        # 3. Prepare output file and run evaluation
        output_file = get_output_filepath(cfg)
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        cprint(f"Output will be saved to: {output_file}", "cyan")

        model.evaluate(data, output_file, cfg.dataset.images_root_dir)
        cprint(f"\nEvaluation finished successfully!", "green", attrs=["bold"])
        cprint(f"Results saved to {output_file}", "green")

    except Exception as e:
        cprint(f"\nFATAL ERROR during execution: {str(e)}", "red", attrs=["bold"])
        raise

if __name__ == "__main__":
    main()