import os

# Path to the main folder containing all tables together
# main_dir = "data/filtered_data/filtered_tables"
# main_dir = "data/processed/tables"
main_dir = "data/benchmark_data/arxiv/tables"

# Keywords to identify each dataset
datasets = ["arxiv", "wikisql", "finqa"]

# Initialize counts
counts = {name: 0 for name in datasets}

# Iterate through all files in the directory
for filename in os.listdir(main_dir):
    if os.path.isfile(os.path.join(main_dir, filename)):
        for name in datasets:
            if name.lower() in filename.lower():
                counts[name] += 1

# Print the result
for name in datasets:
    print(f"{name.upper()} tables: {counts[name]}")
