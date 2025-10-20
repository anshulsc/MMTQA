import argparse
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    """Configuration for the VLLM model and sampling."""
    name: str = "qwen"
    model_path: str = "Qwen/Qwen2-VL-7B-Instruct"
    tensor_parallel_size: int = 1
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90
    temperature: float = 0.0
    max_new_tokens: int = 1024

@dataclass
class DatasetConfig:
    """Configuration for the dataset to be evaluated."""
    name: str = "multitableqa"
    data_file: str = "data/dataset_en.jsonl"
    images_root_dir: str = "data/images"
    image_type: str = "noise"  # 'clean' or 'noise'
    lang_code: str = "en"      # e.g., 'en', 'es', or 'default'
    resolution: int | None = None

@dataclass
class PromptConfig:
    """Configuration for the prompt version."""
    name: str = "default_visual_qa"

@dataclass
class MainConfig:
    """Top-level configuration container."""
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    output_file_template: str = "data/processed/evaluation_results/{model_name}_{dataset_name}_{image_type}"

def get_config() -> MainConfig:
    """
    Parses command-line arguments to override default configurations.
    This function replaces the role of Hydra.
    """
    parser = argparse.ArgumentParser(description="Run VLLM evaluation for MultiTableQA.")
    
    # --- Add arguments to override config values ---
    parser.add_argument("--model_name", type=str, help="Name of the model to use (e.g., qwen, gemma).")
    parser.add_argument("--model_path", type=str, help="Path to the model checkpoint.")
    parser.add_argument("--data_file", type=str, help="Path to the .jsonl data file.")
    parser.add_argument("--images_root_dir", type=str, help="Path to the root directory of images.")
    parser.add_argument("--image_type", type=str, choices=['clean', 'noise'], help="Image type to evaluate.")
    parser.add_argument("--lang_code", type=str, help="Language code to filter by (e.g., 'en', 'es', 'default').")
    
    args = parser.parse_args()
    
    # Create a default config object
    cfg = MainConfig()
    
    # Override defaults with any provided command-line arguments
    if args.model_name:
        cfg.model.name = args.model_name
    if args.model_path:
        cfg.model.model_path = args.model_path
    if args.data_file:
        cfg.dataset.data_file = args.data_file
    if args.images_root_dir:
        cfg.dataset.images_root_dir = args.images_root_dir
    if args.image_type:
        cfg.dataset.image_type = args.image_type
    if args.lang_code:
        cfg.dataset.lang_code = args.lang_code
        
    return cfg