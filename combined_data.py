import json

# Paths to your datasets
real_dataset_file = "C:/Users/HP VICTUS/MMTQA/dataset_en.jsonl"    # gold/reference
demo_predictions_file = "demo_dataset.jsonl"  # predicted responses
combined_file = "combined_eval.jsonl"

# Load real dataset (gold)
gold_data = {}
with open(real_dataset_file, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        gold_data[item["question_id"]] = item

# Load demo predictions
demo_data = {}
with open(demo_predictions_file, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        demo_data[item["question_id"]] = item

# Merge datasets with 'type' field
with open(combined_file, "w", encoding="utf-8") as f:
    for qid, gold_item in gold_data.items():
        # Determine type based on images
        if gold_item.get("image_clean"):
            item_type = "clean"
        elif gold_item.get("images_noise") and len(gold_item.get("images_noise")) > 0:
            item_type = "noise"
        else:
            item_type = None  # fallback if no image info

        combined_item = {
            "question_id": gold_item.get("question_id"),
            "table_id": gold_item.get("table_id"),
            "language": gold_item.get("language"),
            "question_type": gold_item.get("question_type"),
            "question": gold_item.get("question"),
            "golden_answer": gold_item.get("answer"),         # golden answer
            "response": demo_data.get(qid, {}).get("response", []),  # predicted response
            "evidence_cells": gold_item.get("evidence_cells"),
            "reasoning_category": gold_item.get("reasoning_category"),
            "image_clean": gold_item.get("image_clean"),
            "images_noise": gold_item.get("images_noise"),
            "type": item_type                                  # demo field
        }
        f.write(json.dumps(combined_item, ensure_ascii=False) + "\n")

print(f"✅ Combined evaluation JSONL created: {combined_file} ({len(gold_data)} entries)")
