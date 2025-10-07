import json
import os
import re
from pathlib import Path
from collections import Counter
import statistics

def count_english_chars(text):
    """Count only English alphabetic characters (a-z, A-Z)"""
    if text is None or (isinstance(text, float) and str(text) == 'nan'):
        return 0
    return len(re.findall(r'[a-zA-Z]', str(text)))

def process_json_file(filepath):
    """Process a single JSON file and return character count"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_chars = 0
        
        # Extract data from the JSON structure
        if 'data' in data:
            for row in data['data']:
                for cell in row:
                    total_chars += count_english_chars(cell)
        
        return total_chars
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return 0

def calculate_mode(counts):
    """Calculate mode (most frequent value) from a list of counts"""
    if not counts:
        return None
    
    counter = Counter(counts)
    max_frequency = max(counter.values())
    
    # Get all values with the maximum frequency
    modes = [count for count, freq in counter.items() if freq == max_frequency]
    
    # Return the smallest mode if there are multiple modes
    return min(modes) if modes else None

def calculate_median(counts):
    """Calculate median from a list of counts"""
    if not counts:
        return None
    return statistics.median(counts)

def calculate_average_chars(directory_path, output_file='character_stats.json'):
    """Calculate average and mode English character count across all JSON files"""
    json_files = list(Path(directory_path).glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {directory_path}")
        return
    
    file_counts = {}
    total_chars = 0
    char_counts_list = []
    
    for filepath in json_files:
        char_count = process_json_file(filepath)
        file_counts[filepath.name] = char_count
        char_counts_list.append(char_count)
        total_chars += char_count
        print(f"{filepath.name}: {char_count} characters")
    
    avg_chars = total_chars / len(json_files) if json_files else 0
    mode_chars = calculate_mode(char_counts_list)
    median_chars = calculate_median(char_counts_list)
    
    # Create statistics dictionary
    stats = {
        "summary": {
            "total_files_processed": len(json_files),
            "total_english_characters": total_chars,
            "average_characters_per_file": round(avg_chars, 2),
            "median_characters_per_file": median_chars,
            "mode_characters_per_file": mode_chars
        },
        "individual_files": file_counts
    }
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*50}")
    print(f"Total files processed: {len(json_files)}")
    print(f"Total English characters: {total_chars}")
    print(f"Average English characters per file: {avg_chars:.2f}")
    print(f"Median English characters per file: {median_chars}")
    print(f"Mode English characters per file: {mode_chars}")
    print(f"Statistics saved to: {output_file}")
    print(f"{'='*50}")
    
    return avg_chars, median_chars, mode_chars, file_counts

# Usage
if __name__ == "__main__":
    # Replace with your directory path
    directory = "data/processed/tables"  # Current directory
    
    # Specify output file name (optional)
    output_file = "src/filtering/character_stats.json"
    
    # Or specify custom paths:
    # directory = "/path/to/your/json/files"
    # output_file = "/path/to/output/stats.json"
    
    calculate_average_chars(directory, output_file)


# import json
# import os
# import re
# from pathlib import Path
# from collections import Counter

# def count_english_chars(text):
#     """Count only English alphabetic characters (a-z, A-Z)"""
#     if text is None or (isinstance(text, float) and str(text) == 'nan'):
#         return 0
#     return len(re.findall(r'[a-zA-Z]', str(text)))

# def process_json_file(filepath):
#     """Process a single JSON file and return character count"""
#     try:
#         with open(filepath, 'r', encoding='utf-8') as f:
#             data = json.load(f)
        
#         total_chars = 0
        
#         # Extract data from the JSON structure
#         if 'data' in data:
#             for row in data['data']:
#                 for cell in row:
#                     total_chars += count_english_chars(cell)
        
#         return total_chars
    
#     except Exception as e:
#         print(f"Error processing {filepath}: {e}")
#         return 0
    
# def calculate_mode(counts):
#     """Calculate mode (most frequent value) from a list of counts"""
#     if not counts:
#         return None
    
#     counter = Counter(counts)
#     max_frequency = max(counter.values())
#     # Get all values with the maximum frequency
#     modes = [count for count, freq in counter.items() if freq == max_frequency]
    
#     # Return the smallest mode if there are multiple modes
#     return min(modes) if modes else None

# def calculate_average_chars(directory_path, output_file='character_stats.json'):
#     """Calculate average English character count across all JSON files"""
#     json_files = list(Path(directory_path).glob('*.json'))
    
#     if not json_files:
#         print(f"No JSON files found in {directory_path}")
#         return
    
#     file_counts = {}
#     total_chars = 0
#     char_counts_list = []
    
#     for filepath in json_files:
#         char_count = process_json_file(filepath)
#         file_counts[filepath.name] = char_count
#         total_chars += char_count
#         print(f"{filepath.name}: {char_count} characters")
    
#     avg_chars = total_chars / len(json_files) if json_files else 0
#     mode_chars = calculate_mode(char_counts_list)
    
#     # Create statistics dictionary
#     stats = {
#         "summary": {
#             "total_files_processed": len(json_files),
#             "total_english_characters": total_chars,
#             "average_characters_per_file": round(avg_chars, 2),
#             "mode_characters_per_file": mode_chars
#         },
#         "individual_files": file_counts
#     }
    
#     # Save to JSON file
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(stats, f, indent=2, ensure_ascii=False)
    
#     print(f"\n{'='*50}")
#     print(f"Total files processed: {len(json_files)}")
#     print(f"Total English characters: {total_chars}")
#     print(f"Average English characters per file: {avg_chars:.2f}")
#     print(f"Mode English characters per file: {mode_chars}")
#     print(f"Statistics saved to: {output_file}")
#     print(f"{'='*50}")
    
#     return avg_chars, file_counts

# # Usage
# if __name__ == "__main__":
#     # Replace with your directory path
#     directory = "data/processed/tables"  # Current directory
    
#     # Specify output file name (optional)
#     output_file = "src/filtering/character_stats.json"
    
#     # Or specify custom paths:
#     # directory = "/path/to/your/json/files"
#     # output_file = "/path/to/output/stats.json"
    
#     calculate_average_chars(directory, output_file)