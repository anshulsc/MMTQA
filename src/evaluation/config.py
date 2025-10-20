import argparse
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    name: str = "gemma3"
    model_path: str = "google/gemma-3-12b-it""
    tensor_parallel_size: int = 1
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90
    temperature: float = 0.0
    max_new_tokens: int = 512
    batch_size: int = 8  

@dataclass
class DatasetConfig:
    name: str = "multitableqa"
    data_file: str = "/home/anshulsc/links/scratch/ML_VQA_Tab/data/dataset_en.jsonl"
    images_root_dir: str = "/home/anshulsc/links/scratch/ML_VQA_Tab/images/"
    image_type: str = "clean"  # 'clean' or 'noise'
    lang_code: str = "en"      # e.g., 'en', 'es', or 'default'
    resolution: int | None = None

@dataclass
class PromptConfig:
    name: str = "default_visual_qa"

@dataclass
class MainConfig:
    """Top-level configuration container."""
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    output_file_template: str = "data/processed/evaluation_results/{model_name}_{dataset_name}_{image_type}"
    no_batch: bool = False  # Flag to disable batch processing

def get_config() -> MainConfig:
    """
    Parses command-line arguments to override default configurations.
    This function replaces the role of Hydra.
    """
    parser = argparse.ArgumentParser(description="Run VLLM evaluation for MultiTableQA.")
    
    # --- Model arguments ---
    parser.add_argument("--model_name", type=str, help="Name of the model to use (e.g., qwen, gemma).")
    parser.add_argument("--model_path", type=str, help="Path to the model checkpoint.")
    parser.add_argument("--batch_size", type=int, help="Batch size for inference (default: 8).")
    
    # --- Dataset arguments ---
    parser.add_argument("--data_file", type=str, help="Path to the .jsonl data file.")
    parser.add_argument("--images_root_dir", type=str, help="Path to the root directory of images.")
    parser.add_argument("--image_type", type=str, choices=['clean', 'noise'], help="Image type to evaluate.")
    parser.add_argument("--lang_code", type=str, help="Language code to filter by (e.g., 'en', 'es', 'default').")
    
    # --- Evaluation arguments ---
    parser.add_argument("--no_batch", action='store_true', help="Disable batch processing (use single-item mode).")
    
    args = parser.parse_args()
    
    # Create a default config object
    cfg = MainConfig()
    
    # Override defaults with any provided command-line arguments
    if args.model_name:
        cfg.model.name = args.model_name
    if args.model_path:
        cfg.model.model_path = args.model_path
    if args.batch_size:
        cfg.model.batch_size = args.batch_size
    if args.data_file:
        cfg.dataset.data_file = args.data_file
    if args.images_root_dir:
        cfg.dataset.images_root_dir = args.images_root_dir
    if args.image_type:
        cfg.dataset.image_type = args.image_type
    if args.lang_code:
        cfg.dataset.lang_code = args.lang_code
    if args.no_batch:
        cfg.no_batch = True
        
    return cfg