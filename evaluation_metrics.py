import os
import json
import argparse
import re
from collections import Counter
from typing import Union, List
import tqdm
#import wandb
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# -------------------------
# Colors for console printing
# -------------------------
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

# -------------------------
# Normalization Functions
# -------------------------

def normalize_item(item: Union[str, int, float, list]) -> Union[str, int, float, list]:
    """
    Normalize an individual item (str/int/float/list)
    - Strings: lowercased, stripped, numbers cleaned
    - Floats: converted to int if integer
    """
    if isinstance(item, list):
        return [normalize_item(e) for e in item]
    if isinstance(item, (int, float)):
        return int(item) if isinstance(item, float) and item.is_integer() else item
    if isinstance(item, str):
        clean_str = item.replace(",", "").strip()
        if clean_str.replace('.', '', 1).isdigit():
            num = float(clean_str)
            return int(num) if num.is_integer() else round(num, 6)
        return clean_str.lower()
    return item

def normalize_element(element: Union[str, int, float, list, dict]) -> Union[str, int, float, list]:
    """
    Recursively normalize elements including dicts and lists
    """
    if isinstance(element, dict):
        values = [element[key] for key in element.keys()]
        return normalize_element(values)
    if isinstance(element, list):
        return [normalize_element(e) for e in element]
    if isinstance(element, str):
        # remove potential JSON artifacts
        element = re.sub(r'[{}]|[\w-]+:', '', element)
    return normalize_item(element)

def structure_to_string(data: Union[str, list, dict]) -> str:
    """
    Converts any nested data structure (list/dict/JSON string) into a flat, normalized string
    """
    # If input is JSON string, try to load it
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            # fallback: extract words and numbers
            numbers = re.findall(r'\b\d+\b', data)
            words = re.findall(r'\b[a-zA-Z]+\b', data)
            data = numbers + words

    if isinstance(data, dict):
        data = data.get("data", data)  # if key 'data' exists, take it

    normalized = normalize_element(data)

    # flatten nested lists
    def flatten(items):
        result = []
        for item in items if isinstance(items, list) else [items]:
            if isinstance(item, list):
                result.extend(flatten(item))
            else:
                result.append(str(item))
        return result

    flattened = flatten(normalized)
    return " ".join(flattened)

# -------------------------
# Evaluation Metrics
# -------------------------

def compute_exact_match(prediction: str, truth: str) -> int:
    """
    EM = 1 if prediction == truth else 0
    """
    return int(prediction == truth)

def compute_f1(prediction: str, truth: str) -> float:
    """
    F1 score = 2 * (precision * recall) / (precision + recall)
    Token-level overlap between prediction and truth
    """
    pred_tokens = prediction.split()
    truth_tokens = truth.split()
    
    if not pred_tokens or not truth_tokens:
        return int(pred_tokens == truth_tokens)

    common = Counter(pred_tokens) & Counter(truth_tokens)
    overlap = sum(common.values())
    
    if overlap == 0:
        return 0.0
    
    precision = overlap / len(pred_tokens)
    recall = overlap / len(truth_tokens)
    
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

def compute_adaptive_bleu(prediction: str, references: List[str]) -> float:
    """
    Adaptive BLEU as per the paper you provided:
    - Short text (L <= 3): BLEU-1 only
    - Medium text (4 <= L <= 7): avg(BLEU-1, BLEU-2)
    - Long text (L >= 8): avg(BLEU-1, BLEU-2, BLEU-4)
    """
    smoothing = SmoothingFunction().method1
    pred_tokens = prediction.split()
    L = len(pred_tokens)
    ref_tokens_list = [ref.split() for ref in references]

    if L == 0 or not ref_tokens_list:
        return 0.0

    bleu1 = sentence_bleu(ref_tokens_list, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = sentence_bleu(ref_tokens_list, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu4 = sentence_bleu(ref_tokens_list, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

    if L <= 3:
        return bleu1
    elif 4 <= L <= 7:
        return (bleu1 + bleu2) / 2
    else:
        return (bleu1 + bleu2 + bleu4) / 3

# -------------------------
# Main Processing Loop
# -------------------------

def process_jsonl(input_file: str) -> dict:
    """
    Process each JSONL line:
    1. Normalize golden_answer and response
    2. Compute EM, F1, Adaptive BLEU
    3. Track skipped samples (errors)
    4. Aggregate overall metrics
    """
    results = []
    total = 0
    total_skipped = 0
    parse_errors = 0
    cuda_errors = 0
    table_errors = 0

    em_sum = 0
    f1_sum = 0
    bleu_sum = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm.tqdm(f, desc=f"{Colors.OKBLUE}Processing entries{Colors.RESET}"):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue

            # Check for special skipped cases
            response_raw = entry.get("response", "")
            if isinstance(response_raw, str) and any(err in response_raw for err in ["CUDA out of memory", "Table too large"]):
                total_skipped += 1
                if "CUDA out of memory" in response_raw:
                    cuda_errors += 1
                if "Table too large" in response_raw:
                    table_errors += 1
                continue

            total += 1

            # Convert to flat normalized strings
            gold_str = structure_to_string(entry.get("golden_answer", {})).lower()
            model_str = structure_to_string(entry.get("response", {})).lower()

            # Compute metrics
            em = compute_exact_match(model_str, gold_str)
            f1 = compute_f1(model_str, gold_str)
            bleu = compute_adaptive_bleu(model_str, [gold_str])

            # Accumulate sums
            em_sum += em
            f1_sum += f1
            bleu_sum += bleu

            # Store per-sample results
            results.append({
                "question_id": entry.get("question_id"),
                "question": entry.get("question"),
                "golden_answer": entry.get("golden_answer"),
                "model_response": entry.get("response"),
                "exact_match": em,
                "f1_score": f1,
                "bleu_score": bleu,
                "skipped": False
            })

    # Overall metrics
    overall_metrics = {
        "exact_match": em_sum / total if total > 0 else 0,
        "f1_score": f1_sum / total if total > 0 else 0,
        "bleu_score": bleu_sum / total if total > 0 else 0,
        "processed_samples": total,
        "skipped_samples": total_skipped,
        "parse_errors": parse_errors,
        "cuda_errors": cuda_errors,
        "table_errors": table_errors,
        "total_samples": total + total_skipped + parse_errors
    }

    return {"overall_metrics": overall_metrics, "per_sample_results": results}

# -------------------------
# Main CLI
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate QA dataset with EM, F1, and Adaptive BLEU")
    parser.add_argument("--input-file", required=True, help="Path to combined JSONL")
    parser.add_argument("--output-file", required=True, help="Path to save evaluation metrics")
    parser.add_argument("--wandb-project", help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", help="Weights & Biases entity")
    parser.add_argument("--wandb-run-name", help="Weights & Biases run name")
    args = parser.parse_args()

    # Initialize W&B
    if args.wandb_project:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name, config=vars(args))

    print(f"{Colors.OKBLUE}Processing results from:{Colors.RESET} {Colors.BOLD}{args.input_file}{Colors.RESET}")
    metrics = process_jsonl(args.input_file)

    # Save metrics to JSON
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if args.wandb_project:
        wandb.log(metrics["overall_metrics"])

    # Print summary
    print(f"\n{Colors.OKGREEN}Evaluation results saved to:{Colors.RESET} {Colors.OKCYAN}{args.output_file}{Colors.RESET}")
    print(f"{Colors.OKGREEN}Processed Samples:{Colors.RESET} {Colors.OKBLUE}{metrics['overall_metrics']['processed_samples']}{Colors.RESET}")
    print(f"{Colors.WARNING}Skipped Samples:{Colors.RESET} {Colors.OKBLUE}{metrics['overall_metrics']['skipped_samples']}{Colors.RESET}")
    print(f"{Colors.FAIL}Parse Errors:{Colors.RESET} {Colors.OKBLUE}{metrics['overall_metrics']['parse_errors']}{Colors.RESET}")
    print(f"{Colors.OKGREEN}Exact Match: {metrics['overall_metrics']['exact_match']*100:.2f}%")
    print(f"{Colors.OKGREEN}Average F1: {metrics['overall_metrics']['f1_score']*100:.2f}%")
    print(f"{Colors.OKGREEN}Average Adaptive BLEU: {metrics['overall_metrics']['bleu_score']*100:.2f}%")

if __name__ == "__main__":
    main()
