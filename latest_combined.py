import json
from pathlib import Path
import re

# -------------------------------
# Input and Output Paths
# -------------------------------
real_dataset_file = Path("C:/Users/HP VICTUS/MMTQA/dataset_combined_final.jsonl")
demo_predictions_file = Path("C:/Users/HP VICTUS/MMTQA/data/processed/evaluation_results_new/Qwen_Qwen2.5-VL-72B-Instruct_multitableqa_clean_default_20251022_022301.jsonl")
combined_file = Path("Qwen2.5-VL-72B-clean.jsonl")
missing_file = Path("missing_predictions.jsonl")

# -------------------------------
# Step 1: Load Gold Data
# -------------------------------
gold_data = {}
with open(real_dataset_file, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        key = (item["question_id"], item["language"])
        gold_data[key] = item

# -------------------------------
# Step 2: Load Prediction Data & Extract Language
# -------------------------------
demo_data = {}
# Regex to capture everything before last _clean or _noiseX (handles multi-underscore language codes)
pattern = re.compile(r"^(.*?)(?:_clean|_noise\d*)\.")

with open(demo_predictions_file, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        qid = item.get("question_id")
        image_filename = item.get("image_filename", "")
        if not qid or not image_filename:
            continue
        match = pattern.match(image_filename)
        lang = match.group(1) if match else "unknown"
        key = (qid, lang)
        demo_data[key] = item

# -------------------------------
# Step 3: Combine Data
# -------------------------------
missing_predictions = []
combined = []

with open(combined_file, "w", encoding="utf-8") as f_out:
    for key, gold_item in gold_data.items():
        qid, lang = key
        pred_item = demo_data.get(key, {})

        if not pred_item:
            missing_predictions.append({"question_id": qid, "language": lang})

        # Determine type and variant
        if gold_item.get("image_clean"):
            item_type = "clean"
            variant = "clean"
            image_filename = gold_item.get("image_clean")
        elif gold_item.get("images_noise"):
            item_type = "noise"
            variant = "noise"
            image_filename = gold_item["images_noise"][0] if gold_item["images_noise"] else None
        else:
            item_type = variant = "unknown"
            image_filename = None

        combined_item = {
            "question_id": gold_item.get("question_id"),
            "table_id": gold_item.get("table_id"),
            "language": gold_item.get("language"),
            "question_type": gold_item.get("question_type"),
            "question": gold_item.get("question"),
            "golden_answer": gold_item.get("answer"),
            "response": pred_item.get("model_response", []),
            "evidence_cells": gold_item.get("evidence_cells"),
            "reasoning_category": gold_item.get("reasoning_category"),
            "image_clean": gold_item.get("image_clean"),
            "images_noise": gold_item.get("images_noise"),
            "image_filename": image_filename,
            "type": item_type,
            "variant": variant,
            "model_name": pred_item.get("model_name"),
            "is_correct": pred_item.get("is_correct", False)
        }

        f_out.write(json.dumps(combined_item, ensure_ascii=False) + "\n")
        combined.append(combined_item)

# -------------------------------
# Step 4: Save Missing Predictions
# -------------------------------
with open(missing_file, "w", encoding="utf-8") as f_miss:
    for item in missing_predictions:
        f_miss.write(json.dumps(item, ensure_ascii=False) + "\n")

# -------------------------------
# Step 5: Summary
# -------------------------------
print(f"✅ Combined JSONL created: {combined_file}")
print(f"Total Gold Items: {len(gold_data)}")
print(f"Total Predictions (matched): {len(demo_data)}")
print(f"Missing Predictions: {len(missing_predictions)}")
print(f"Final Combined Entries: {len(combined)}")
print(f"Missing predictions saved to: {missing_file}")
