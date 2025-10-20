import json
import signal
import torch
from pathlib import Path
from PIL import Image
from termcolor import cprint
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from pydantic import ValidationError

from ..prompts import Response
from ..config import MainConfig # <-- IMPORT FROM PYTHON CONFIG

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Inference timed out!")

class BaseModel:
    def __init__(self, cfg: MainConfig): # <-- USE MainConfig TYPE
        self.cfg = cfg
        self.model = self.load_model()
        self.processor = self.load_processor()
        self.resolution = cfg.dataset.resolution

    def load_model(self):
        cprint(f"Loading VLLM engine for model: {self.cfg.model.model_path}", "yellow")
        cprint("This may take a few minutes...", "yellow")
        
        llm = LLM(
            model=self.cfg.model.model_path,
            tensor_parallel_size=self.cfg.model.tensor_parallel_size,
            max_model_len=self.cfg.model.max_model_len,
            gpu_memory_utilization=self.cfg.model.gpu_memory_utilization,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 10}
        )
        cprint("VLLM Engine loaded successfully.", "green")
        return llm

    def load_processor(self):
        raise NotImplementedError("Each model must implement its own processor loader.")

    def prepare_input(self, data_row, images_dir):
        raise NotImplementedError("Each model must implement its own input preparer.")

    def _create_vllm_sampling_params(self) -> SamplingParams:
        try:
            guided_params = GuidedDecodingParams(json_schema=Response.model_json_schema())
        except Exception as e:
            cprint(f"Warning: Could not create guided decoding params. Error: {e}", "red")
            guided_params = None

        return SamplingParams(
            temperature=self.cfg.model.temperature,
            max_tokens=self.cfg.model.max_new_tokens,
            guided_decoding=guided_params,
        )

    def generate_response(self, inputs: dict) -> str:
        sampling_params = self._create_vllm_sampling_params()
        outputs = self.model.generate(
            prompt=inputs["prompt"],
            multi_modal_data=inputs["multi_modal_data"],
            sampling_params=sampling_params
        )
        return outputs[0].outputs[0].text

    def _parse_model_response(self, response_str: str) -> list:
        try:
            parsed_response = Response.model_validate_json(response_str)
            return parsed_response.data
        except (json.JSONDecodeError, ValidationError) as e:
            cprint(f"\n[WARN] Failed to parse model output as valid JSON: {response_str}. Error: {e}", "yellow")
            return [["PARSING_ERROR", str(e)]]
        except Exception as e:
            cprint(f"\n[WARN] An unexpected error occurred during parsing: {e}", "yellow")
            return [["UNEXPECTED_PARSING_ERROR", str(e)]]

    def evaluate(self, data: list, output_file: str, images_dir: str):
        cprint(f"Starting evaluation on {len(data)} instances...", "cyan")
        with open(output_file, "a+", encoding="utf-8") as out_file:
            for i, row in enumerate(tqdm(data, desc="Evaluating")):
                try:
                    inputs = self.prepare_input(row, images_dir)
                    raw_response_str = self.generate_response_with_timeout(inputs, timeout=300)
                    parsed_response_data = self._parse_model_response(raw_response_str)
                    result = self.create_result_dict(row, parsed_response_data)
                except Exception as e:
                    result = self.handle_exception(row, e)
                
                out_file.write(json.dumps(result) + "\n")
                if (i + 1) % 20 == 0:
                    out_file.flush()

    def generate_response_with_timeout(self, inputs, timeout=120):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        try:
            result = self.generate_response(inputs)
        except TimeoutException:
            raise TimeoutException(f"Inference timed out after {timeout} seconds")
        finally:
            signal.alarm(0)
        return result

    def _resize_image(self, image: Image.Image) -> Image.Image:
        if self.resolution and isinstance(self.resolution, int):
            return image.resize((self.resolution, self.resolution), Image.LANCZOS)
        return image

    def create_result_dict(self, row, parsed_data: list):
        return {
            "question_id": row.get("question_id"),
            "question": row.get("question"),
            "golden_answer": row.get("golden_answer"),
            "image_filename": row.get("image_filename"),
            "reasoning_category": row.get("reasoning_category"),
            "model_response": parsed_data,
            "model_name": self.cfg.model.model_path,
        }

    def handle_exception(self, row, e):
        cprint(f"\nError processing question {row['question_id']}: {e}", "red")
        error_message = f"Error: {e}"
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            error_message = "CUDA out of memory. Skipping."
        elif isinstance(e, TimeoutException):
            error_message = "Inference timed out."
        return self.create_result_dict(row, [["GENERATION_ERROR", error_message]])