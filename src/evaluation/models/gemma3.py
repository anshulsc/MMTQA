from pathlib import Path
from PIL import Image
from transformers import AutoProcessor
from termcolor import cprint

from src.evaluation.models.base_model import BaseModel
from src.evaluation.prompts import VISUAL_TABLE_QA_SYSTEM_PROMPT
from src.evaluation.config import MainConfig

class Gemma3Model(BaseModel):
    def __init__(self, cfg: MainConfig):
        super().__init__(cfg)

    def load_processor(self):
        cprint(f"Loading processor for: {self.cfg.model.model_path}", "blue")
        return AutoProcessor.from_pretrained(
            self.cfg.model.model_path, 
            trust_remote_code=True,
            padding_side="left"
        )

    def prepare_input(self, data_row: dict, images_dir: str) -> dict:
        table_id = data_row['table_id']
        image_type = self.cfg.dataset.image_type
        image_filename = data_row['image_filename']
        
        image_path = Path(images_dir) / table_id / image_type / image_filename
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        image = Image.open(image_path).convert("RGB")
        resized_image = self._resize_image(image)

        # Gemma3 uses a specific message format with system and user roles
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": VISUAL_TABLE_QA_SYSTEM_PROMPT}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": resized_image},
                    {"type": "text", "text": f"Question: {data_row['question']}"}
                ]
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            return_dict=False
        )

        return {
            "prompt": inputs,
            "multi_modal_data": {"image": [resized_image]}
        }