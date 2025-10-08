import json
import os
import numpy as np
from pathlib import Path

def has_missing_data(file_path):
    """
    Check if a JSON file contains missing data (NaN values) in its table.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        True if file has missing data, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if 'data' key exists
        if 'data' not in data:
            return False
        
        # Check for NaN values in the data
        for row in data['data']:
            for cell in row:
                # Check for None, NaN string, or actual NaN
                if cell is None or cell == 'NaN' or (isinstance(cell, float) and np.isnan(cell)):
                    return True
        
        return False
    
    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        print(f"Error reading {file_path}: {e}")
        return False

def extract_hash_from_filename(file_path):
    """
    Extract the hash identifier from a filename.
    
    Args:
        file_path: Path object of the file
        
    Returns:
        Hash string (e.g., '0b8207bf5f' from 'arxiv_0b8207bf5f.json')
    """
    filename = file_path.stem  # Get filename without extension
    # Assuming format is 'source_hash', extract the hash part
    parts = filename.split('_')
    if len(parts) >= 2:
        return '_'.join(parts[1:])  # Return everything after first underscore
    return filename

def find_related_files(data_file, metadata_dir):
    """
    Find the metadata file that corresponds to a data file.
    
    Args:
        data_file: Path to the data file
        metadata_dir: Path to the metadata directory
        
    Returns:
        Path to metadata file if found, None otherwise
    """
    file_hash = extract_hash_from_filename(data_file)
    metadata_dir = Path(metadata_dir)
    
    # Look for matching metadata file with same hash
    for metadata_file in metadata_dir.glob('**/*.json'):
        if file_hash in metadata_file.stem:
            return metadata_file
    
    return None

def remove_files_with_missing_data(data_directory, metadata_directory=None, dry_run=True):
    """
    Remove JSON files that contain missing data and their corresponding metadata files.
    
    Args:
        data_directory: Directory containing data JSON files
        metadata_directory: Directory containing metadata JSON files (optional)
        dry_run: If True, only list files without deleting them
    """
    data_directory = Path(data_directory)
    files_to_remove = []
    
    # Find all JSON files in data directory
    json_files = list(data_directory.glob('**/*.json'))
    
    print(f"Scanning {len(json_files)} JSON files...")
    
    # Check each file for missing data
    for file_path in json_files:
        if has_missing_data(file_path):
            file_pair = {'data': file_path, 'metadata': None}
            
            # Find corresponding metadata file if metadata directory is provided
            if metadata_directory:
                metadata_file = find_related_files(file_path, metadata_directory)
                if metadata_file:
                    file_pair['metadata'] = metadata_file
            
            files_to_remove.append(file_pair)
    
    print(f"\nFound {len(files_to_remove)} files with missing data:")
    for pair in files_to_remove:
        print(f"\n  Data file: {pair['data']}")
        if pair['metadata']:
            print(f"  Metadata file: {pair['metadata']}")
        else:
            print(f"  Metadata file: Not found")
    
    if not dry_run:
        print("\n" + "="*50)
        print("DELETING FILES...")
        print("="*50)
        deleted_count = 0
        
        for pair in files_to_remove:
            # Delete data file
            try:
                os.remove(pair['data'])
                print(f"✓ Deleted data: {pair['data']}")
                deleted_count += 1
            except Exception as e:
                print(f"✗ Error deleting {pair['data']}: {e}")
            
            # Delete metadata file if it exists
            if pair['metadata']:
                try:
                    os.remove(pair['metadata'])
                    print(f"✓ Deleted metadata: {pair['metadata']}")
                    deleted_count += 1
                except Exception as e:
                    print(f"✗ Error deleting {pair['metadata']}: {e}")
        
        print(f"\nTotal files removed: {deleted_count}")
    else:
        print("\n" + "="*50)
        print("DRY RUN MODE - No files were deleted")
        print("="*50)
        print("Set dry_run=False to actually delete these files.")

# Example usage
if __name__ == "__main__":
    # Specify your directory paths
    data_dir = "data/processed/tables"  # Directory containing data files
    metadata_dir = "data/processed/metadata"  # Directory containing metadata files
    
    # First run in dry-run mode to see what would be deleted
    print("=== DRY RUN MODE ===\n")
    remove_files_with_missing_data(
        data_directory=data_dir,
        metadata_directory=metadata_dir,
        dry_run=True
    )
    
    # Uncomment the lines below to actually delete the files
    print("\n\n=== DELETING FILES ===\n")
    remove_files_with_missing_data(
        data_directory=data_dir,
        metadata_directory=metadata_dir,
        dry_run=False
    )