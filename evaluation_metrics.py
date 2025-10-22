# import os
# import json
# import argparse
# import re
# from collections import Counter
# from typing import Union, List
# import tqdm
# #import wandb
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# # -------------------------
# # Colors for console printing
# # -------------------------
# class Colors:
#     HEADER = '\033[95m'
#     OKBLUE = '\033[94m'
#     OKCYAN = '\033[96m'
#     OKGREEN = '\033[92m'
#     WARNING = '\033[93m'
#     FAIL = '\033[91m'
#     BOLD = '\033[1m'
#     UNDERLINE = '\033[4m'
#     RESET = '\033[0m'

# # -------------------------
# # Normalization Functions
# # -------------------------

# def normalize_item(item: Union[str, int, float, list]) -> Union[str, int, float, list]:
#     if isinstance(item, list):
#         return [normalize_item(e) for e in item]
#     if isinstance(item, (int, float)):
#         return int(item) if isinstance(item, float) and item.is_integer() else item
#     if isinstance(item, str):
#         clean_str = item.replace(",", "").strip()
#         if clean_str.replace('.', '', 1).isdigit():
#             num = float(clean_str)
#             return int(num) if num.is_integer() else round(num, 6)
#         return clean_str.lower()
#     return item

# def normalize_element(element: Union[str, int, float, list, dict]) -> Union[str, int, float, list]:
#     if isinstance(element, dict):
#         values = [element[key] for key in element.keys()]
#         return normalize_element(values)
#     if isinstance(element, list):
#         return [normalize_element(e) for e in element]
#     if isinstance(element, str):
#         element = re.sub(r'[{}]|[\w-]+:', '', element)
#     return normalize_item(element)

# def structure_to_string(data: Union[str, list, dict]) -> str:
#     if isinstance(data, str):
#         try:
#             data = json.loads(data)
#         except json.JSONDecodeError:
#             numbers = re.findall(r'\b\d+\b', data)
#             words = re.findall(r'\b[a-zA-Z]+\b', data)
#             data = numbers + words

#     if isinstance(data, dict):
#         data = data.get("data", data)

#     normalized = normalize_element(data)

#     def flatten(items):
#         result = []
#         for item in items if isinstance(items, list) else [items]:
#             if isinstance(item, list):
#                 result.extend(flatten(item))
#             else:
#                 result.append(str(item))
#         return result

#     flattened = flatten(normalized)
#     return " ".join(flattened)

# # -------------------------
# # Evaluation Metrics
# # -------------------------

# def compute_exact_match(prediction: str, truth: str) -> int:
#     return int(prediction == truth)

# def compute_f1(prediction: str, truth: str) -> float:
#     pred_tokens = prediction.split()
#     truth_tokens = truth.split()
    
#     if not pred_tokens or not truth_tokens:
#         return int(pred_tokens == truth_tokens)

#     common = Counter(pred_tokens) & Counter(truth_tokens)
#     overlap = sum(common.values())
    
#     if overlap == 0:
#         return 0.0
    
#     precision = overlap / len(pred_tokens)
#     recall = overlap / len(truth_tokens)
    
#     return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

# def compute_adaptive_bleu(prediction: str, references: List[str]) -> float:
#     smoothing = SmoothingFunction().method1
#     pred_tokens = prediction.split()
#     L = len(pred_tokens)
#     ref_tokens_list = [ref.split() for ref in references]

#     if L == 0 or not ref_tokens_list:
#         return 0.0

#     bleu1 = sentence_bleu(ref_tokens_list, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
#     bleu2 = sentence_bleu(ref_tokens_list, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
#     bleu4 = sentence_bleu(ref_tokens_list, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

#     if L <= 3:
#         return bleu1
#     elif 4 <= L <= 7:
#         return (bleu1 + bleu2) / 2
#     else:
#         return (bleu1 + bleu2 + bleu4) / 3

# # -------------------------
# # Main Processing Loop
# # -------------------------

# def process_jsonl(input_file: str) -> dict:
#     results = []
#     total = 0
#     total_skipped = 0
#     parse_errors = 0
#     cuda_errors = 0
#     table_errors = 0

#     em_sum = 0
#     f1_sum = 0
#     bleu_sum = 0

#     with open(input_file, "r", encoding="utf-8") as f:
#         for line in tqdm.tqdm(f, desc=f"{Colors.OKBLUE}Processing entries{Colors.RESET}"):
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 entry = json.loads(line)
#             except json.JSONDecodeError:
#                 parse_errors += 1
#                 continue

#             response_raw = entry.get("response", "")
#             if isinstance(response_raw, str) and any(err in response_raw for err in ["CUDA out of memory", "Table too large"]):
#                 total_skipped += 1
#                 if "CUDA out of memory" in response_raw:
#                     cuda_errors += 1
#                 if "Table too large" in response_raw:
#                     table_errors += 1
#                 continue

#             total += 1

#             gold_str = structure_to_string(entry.get("golden_answer", {})).lower()
#             model_str = structure_to_string(entry.get("response", {})).lower()

#             em = compute_exact_match(model_str, gold_str)
#             f1 = compute_f1(model_str, gold_str)
#             bleu = compute_adaptive_bleu(model_str, [gold_str])

#             em_sum += em
#             f1_sum += f1
#             bleu_sum += bleu

#             # ✅ Added reasoning_category and type (clean/noise)
#             results.append({
#                 "question_id": entry.get("question_id"),
#                 "question": entry.get("question"),
#                 "golden_answer": entry.get("golden_answer"),
#                 "model_response": entry.get("response"),
#                 "reasoning_category": entry.get("reasoning_category", None),
#                 "type": entry.get("type", None),
#                 "exact_match": em,
#                 "f1_score": f1,
#                 "bleu_score": bleu,
#                 "skipped": False
#             })

#     overall_metrics = {
#         "exact_match": em_sum / total if total > 0 else 0,
#         "f1_score": f1_sum / total if total > 0 else 0,
#         "bleu_score": bleu_sum / total if total > 0 else 0,
#         "processed_samples": total,
#         "skipped_samples": total_skipped,
#         "parse_errors": parse_errors,
#         "cuda_errors": cuda_errors,
#         "table_errors": table_errors,
#         "total_samples": total + total_skipped + parse_errors
#     }

#     return {"overall_metrics": overall_metrics, "per_sample_results": results}

# # -------------------------
# # Main CLI
# # -------------------------

# def main():
#     parser = argparse.ArgumentParser(description="Evaluate QA dataset with EM, F1, and Adaptive BLEU")
#     parser.add_argument("--input-file", required=True, help="Path to combined JSONL")
#     parser.add_argument("--output-file", required=True, help="Path to save evaluation metrics")
#     parser.add_argument("--wandb-project", help="Weights & Biases project name")
#     parser.add_argument("--wandb-entity", help="Weights & Biases entity")
#     parser.add_argument("--wandb-run-name", help="Weights & Biases run name")
#     args = parser.parse_args()

#     if args.wandb_project:
#         wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name, config=vars(args))

#     print(f"{Colors.OKBLUE}Processing results from:{Colors.RESET} {Colors.BOLD}{args.input_file}{Colors.RESET}")
#     metrics = process_jsonl(args.input_file)

#     os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
#     with open(args.output_file, "w", encoding="utf-8") as f:
#         json.dump(metrics, f, indent=2)

#     if args.wandb_project:
#         wandb.log(metrics["overall_metrics"])

#     print(f"\n{Colors.OKGREEN}Evaluation results saved to:{Colors.RESET} {Colors.OKCYAN}{args.output_file}{Colors.RESET}")
#     print(f"{Colors.OKGREEN}Processed Samples:{Colors.RESET} {Colors.OKBLUE}{metrics['overall_metrics']['processed_samples']}{Colors.RESET}")
#     print(f"{Colors.WARNING}Skipped Samples:{Colors.RESET} {Colors.OKBLUE}{metrics['overall_metrics']['skipped_samples']}{Colors.RESET}")
#     print(f"{Colors.FAIL}Parse Errors:{Colors.RESET} {Colors.OKBLUE}{metrics['overall_metrics']['parse_errors']}{Colors.RESET}")
#     print(f"{Colors.OKGREEN}Exact Match: {metrics['overall_metrics']['exact_match']*100:.2f}%")
#     print(f"{Colors.OKGREEN}Average F1: {metrics['overall_metrics']['f1_score']*100:.2f}%")
#     print(f"{Colors.OKGREEN}Average Adaptive BLEU: {metrics['overall_metrics']['bleu_score']*100:.2f}%")

# if __name__ == "__main__":
#     main()




import os
import json
import argparse
import re
from collections import Counter
from typing import Union, List
import tqdm
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

# def normalize_item(item: Union[str, int, float, list]) -> Union[str, int, float, list]:
#     if isinstance(item, list):
#         return [normalize_item(e) for e in item]
#     if isinstance(item, (int, float)):
#         return int(item) if isinstance(item, float) and item.is_integer() else item
#     if isinstance(item, str):
#         clean_str = item.replace(",", "").strip()
#         if clean_str.replace('.', '', 1).isdigit():
#             num = float(clean_str)
#             return int(num) if num.is_integer() else round(num, 6)
#         return clean_str.lower()
#     return item
def normalize_item(item: Union[str, int, float, list]) -> Union[str, int, float, list]:
    if isinstance(item, list):
        return [normalize_item(e) for e in item]
    if isinstance(item, (int, float)):
        return int(item) if isinstance(item, float) and item.is_integer() else item
    if isinstance(item, str):
        # Remove commas and spaces
        clean_str = item.replace(",", "").strip()
        
        # Remove superscripts or Unicode characters in numbers
        clean_str = re.sub(r"[^\d\.\-eE]", "", clean_str)
        
        # Convert to number if possible
        try:
            if clean_str.replace('.', '', 1).isdigit() or re.match(r"^-?\d+(\.\d+)?([eE]-?\d+)?$", clean_str):
                num = float(clean_str)
                return int(num) if num.is_integer() else round(num, 6)
        except ValueError:
            pass
        
        return clean_str.lower()
    return item


def normalize_element(element: Union[str, int, float, list, dict]) -> Union[str, int, float, list]:
    if isinstance(element, dict):
        values = [element[key] for key in element.keys()]
        return normalize_element(values)
    if isinstance(element, list):
        return [normalize_element(e) for e in element]
    if isinstance(element, str):
        element = re.sub(r'[{}]|[\w-]+:', '', element)
    return normalize_item(element)

def structure_to_string(data: Union[str, list, dict]) -> str:
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            numbers = re.findall(r'\b\d+\b', data)
            words = re.findall(r'\b[a-zA-Z]+\b', data)
            data = numbers + words

    if isinstance(data, dict):
        data = data.get("data", data)

    normalized = normalize_element(data)

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
    return int(prediction == truth)

def compute_f1(prediction: str, truth: str) -> float:
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
    results = []
    total = 0
    total_skipped = 0
    parse_errors = 0
    cuda_errors = 0
    table_errors = 0

    em_sum = 0
    f1_sum = 0
    bleu_sum = 0

    # For per-language metrics
    lang_metrics = {}

    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm.tqdm(f, desc="Processing entries"):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue

            response_raw = entry.get("response", "")
            if isinstance(response_raw, str) and any(err in response_raw for err in ["CUDA out of memory", "Table too large"]):
                total_skipped += 1
                if "CUDA out of memory" in response_raw:
                    cuda_errors += 1
                if "Table too large" in response_raw:
                    table_errors += 1
                continue

            total += 1

            gold_str = structure_to_string(entry.get("golden_answer", {})).lower()
            model_str = structure_to_string(entry.get("response", {})).lower()

            em = compute_exact_match(model_str, gold_str)
            f1 = compute_f1(model_str, gold_str)
            bleu = compute_adaptive_bleu(model_str, [gold_str])

            em_sum += em
            f1_sum += f1
            bleu_sum += bleu

            # Per-language aggregation
            lang = entry.get("language", "unknown")
            if lang not in lang_metrics:
                lang_metrics[lang] = {"em_sum": 0, "f1_sum": 0, "bleu_sum": 0, "count": 0}
            lang_metrics[lang]["em_sum"] += em
            lang_metrics[lang]["f1_sum"] += f1
            lang_metrics[lang]["bleu_sum"] += bleu
            lang_metrics[lang]["count"] += 1

            results.append({
                "question_id": entry.get("question_id"),
                "question": entry.get("question"),
                "golden_answer": entry.get("golden_answer"),
                "model_response": entry.get("response"),
                "reasoning_category": entry.get("reasoning_category", None),
                "type": entry.get("type", None),
                "language": lang,
                "exact_match": em,
                "f1_score": f1,
                "bleu_score": bleu,
                "skipped": False
            })

    # Compute per-language averages
    per_language_metrics = {}
    for lang, stats in lang_metrics.items():
        count = stats["count"]
        per_language_metrics[lang] = {
            "exact_match": stats["em_sum"] / count if count else 0,
            "f1_score": stats["f1_sum"] / count if count else 0,
            "bleu_score": stats["bleu_sum"] / count if count else 0,
            "samples": count
        }

    overall_metrics = {
        "exact_match": em_sum / total if total > 0 else 0,
        "f1_score": f1_sum / total if total > 0 else 0,
        "bleu_score": bleu_sum / total if total > 0 else 0,
        "processed_samples": total,
        "skipped_samples": total_skipped,
        "parse_errors": parse_errors,
        "cuda_errors": cuda_errors,
        "table_errors": table_errors,
        "total_samples": total + total_skipped + parse_errors,
        "per_language_metrics": per_language_metrics
    }

    return {"overall_metrics": overall_metrics, "per_sample_results": results}

# -------------------------
# Main CLI
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate QA dataset with EM, F1, and Adaptive BLEU")
    parser.add_argument("--input-file", required=True, help="Path to combined JSONL")
    parser.add_argument("--output-file", required=True, help="Path to save evaluation metrics")
    args = parser.parse_args()

    print(f"{Colors.OKBLUE}Processing results from:{Colors.RESET} {Colors.BOLD}{args.input_file}{Colors.RESET}")
    metrics = process_jsonl(args.input_file)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # -------------------------
    # Print summary
    # -------------------------
    print(f"\n{Colors.OKGREEN}Overall Metrics:{Colors.RESET}")
    print(f"Processed Samples: {metrics['overall_metrics']['processed_samples']}")
    print(f"Skipped Samples: {metrics['overall_metrics']['skipped_samples']}")
    print(f"Exact Match: {metrics['overall_metrics']['exact_match']*100:.2f}%")
    print(f"Average F1: {metrics['overall_metrics']['f1_score']*100:.2f}%")
    print(f"Average Adaptive BLEU: {metrics['overall_metrics']['bleu_score']*100:.2f}%")

    print(f"\n{Colors.OKCYAN}Per-Language Metrics:{Colors.RESET}")
    for lang, vals in metrics["overall_metrics"]["per_language_metrics"].items():
        print(f"{Colors.BOLD}{lang}:{Colors.RESET} EM={vals['exact_match']*100:.2f}%, F1={vals['f1_score']*100:.2f}%, BLEU={vals['bleu_score']*100:.2f}%, Samples={vals['samples']}")

if __name__ == "__main__":
    main()
