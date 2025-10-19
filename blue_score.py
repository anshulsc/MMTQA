# import os
# import json
# import csv
# from collections import defaultdict

# root_dir = "C:/Users/HP VICTUS/MMTQA/data/processed/translation_metadata"
# output_csv = "bleu_scores_summary.csv"

# # Dictionary: { table_id: { lang_code: bleu_score } }
# data_dict = defaultdict(dict)
# all_langs = set()

# # Loop through all tables
# for table_folder in os.listdir(root_dir):
#     table_path = os.path.join(root_dir, table_folder)
#     if not os.path.isdir(table_path):
#         continue

#     # Loop through language JSON files
#     for json_file in os.listdir(table_path):
#         if json_file.endswith(".json"):
#             json_path = os.path.join(table_path, json_file)
#             try:
#                 with open(json_path, "r", encoding="utf-8") as f:
#                     data = json.load(f)

#                 table_id = data.get("table_id", table_folder)
#                 lang_code = data.get("lang_code", os.path.splitext(json_file)[0])
#                 bleu_score = data.get("bleu_score", "")

#                 data_dict[table_id][lang_code] = bleu_score
#                 all_langs.add(lang_code)
#             except Exception as e:
#                 print(f"Error reading {json_path}: {e}")

# # Sort language columns
# all_langs = sorted(all_langs)

# # Write to CSV
# with open(output_csv, "w", newline='', encoding="utf-8") as csvfile:
#     fieldnames = ["table_id"] + all_langs
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()

#     for table_id, lang_scores in data_dict.items():
#         row = {"table_id": table_id}
#         for lang in all_langs:
#             row[lang] = lang_scores.get(lang, "")
#         writer.writerow(row)

# print(f"✅ BLEU score summary created: {output_csv}")
import csv

input_csv = "bleu_scores_summary.csv"
output_csv = "bleu_language_stats.csv"

# Read the input CSV
with open(input_csv, "r", encoding="utf-8") as infile:
    reader = csv.DictReader(infile)
    lang_codes = [col for col in reader.fieldnames if col != "table_id"]

    # Initialize accumulators
    lang_sums = {lang: 0.0 for lang in lang_codes}
    lang_counts = {lang: 0 for lang in lang_codes}

    # Iterate through rows
    for row in reader:
        for lang in lang_codes:
            val = row.get(lang, "")
            if val != "":
                try:
                    val = float(val)
                    lang_sums[lang] += val
                    lang_counts[lang] += 1
                except ValueError:
                    continue

# Compute averages
lang_averages = {lang: (lang_sums[lang] / lang_counts[lang] if lang_counts[lang] > 0 else 0.0)
                 for lang in lang_codes}

# Write output CSV
with open(output_csv, "w", newline='', encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["lang_code", "total_bleu", "avg_bleu"])
    for lang in lang_codes:
        writer.writerow([lang, round(lang_sums[lang], 6), round(lang_averages[lang], 6)])

print(f"✅ Language BLEU statistics saved to: {output_csv}")
