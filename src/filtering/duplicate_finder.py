import json
from collections import defaultdict
from pathlib import Path

def check_duplicates(directory):
    """
    Check for duplicate tables by their existing hashes.
    Tables are named as {category}_hashnumber
    Searches for JSON files in the given directory.
    """
    hash_to_tables = defaultdict(list)
    json_files = list(Path(directory).glob('*.json'))
    
    if not json_files:
        print(f"❌ No JSON files found in directory: {directory}")
        return False, {}
    
    print(f"Found {len(json_files)} JSON files\n")
    print("Checking for duplicate hashes...\n")
    print("-" * 60)
    
    for json_file in json_files:
        table_name = json_file.stem  # Get filename without extension
        
        # Extract hash from table name (everything after the last underscore)
        parts = table_name.rsplit('_', 1)
        if len(parts) == 2:
            hash_value = parts[1]
            hash_to_tables[hash_value].append(table_name)
    
    # Find and report duplicates
    duplicates_found = False
    for hash_value, table_names in hash_to_tables.items():
        if len(table_names) > 1:
            duplicates_found = True
            print(f"\n🔴 DUPLICATE HASH FOUND:")
            print(f"   Hash: {hash_value}")
            print(f"   Tables ({len(table_names)}):")
            for name in table_names:
                print(f"     - {name}")
    
    print("\n" + "-" * 60)
    
    if not duplicates_found:
        print("\n✅ No duplicates found! All hashes are unique.")
    else:
        print(f"\n⚠️  Duplicates detected!")
    
    print(f"\nTotal tables checked: {len(json_files)}")
    print(f"Unique hashes: {len(hash_to_tables)}")
    
    return duplicates_found, hash_to_tables

def main():
    # Specify your directory here
    directory = "data/benchmark_dataset/benchmark_metadata"  # Change this to your directory path
    
    try:
        duplicates_found, hashes = check_duplicates(directory)
        
        # Return exit code for scripting purposes
        return 1 if duplicates_found else 0
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())