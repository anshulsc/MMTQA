import json
import pandas as pd

# ---- Step 1: Load your dataset (replace with your actual dataset path)
data_path = "dataset_en.jsonl"  # can be .json, .jsonl, or .csv

# Load according to your format
if data_path.endswith(".jsonl"):
    data = [json.loads(line) for line in open(data_path, "r", encoding="utf-8")]
elif data_path.endswith(".json"):
    data = json.load(open(data_path, "r", encoding="utf-8"))
elif data_path.endswith(".csv"):
    data = pd.read_csv(data_path).to_dict(orient="records")
else:
    raise ValueError("Unsupported file format! Use .json, .jsonl, or .csv")

# ---- Step 2: Create the new demo JSONL structure
output_records = []

for item in data:
    question_id = item.get("question_id")
    question = item.get("question")
    language = item.get("language")
    answer = item.get("answer")  # this already contains [["..."]] format
    reasoning_category = item.get("reasoning_category")

    # Determine clean/noise type
    if "image_clean" in item and item["image_clean"]:
        img_type = "clean"
    elif "images_noise" in item and item["images_noise"]:
        img_type = "noise"
    else:
        img_type = None

    record = {
        "question_id": question_id,
        "question": question,
        "language": language,
        "response": answer,  # ✅ Keeps [["..."]] intact
        "type": img_type,
        "category": reasoning_category
    }

    output_records.append(record)

# ---- Step 3: Save as demo JSONL
output_path = "demo_dataset.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for r in output_records:
        json.dump(r, f, ensure_ascii=False)
        f.write("\n")

print(f"✅ Demo JSONL created successfully with {len(output_records)} entries → {output_path}")
