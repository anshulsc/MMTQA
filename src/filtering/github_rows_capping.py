import json
import os
import random
from pathlib import Path

def cap_json_rows(directory_path, output_directory=None):
    """
    Process JSON files with 'github_csv' prefix and cap rows between 10-20 randomly.
    Reads from tables/ and metadata/ subdirectories.
    Also updates corresponding metadata JSON files in separate folders.
    
    Args:
        directory_path (str): Path to the directory containing 'tables' and 'metadata' subdirectories
        output_directory (str): Path to save processed files (if None, overwrites original files)
    """
    
    dir_path = Path(directory_path)
    
    if not dir_path.exists():
        print(f"Error: Directory '{directory_path}' does not exist.")
        return
    
    if not dir_path.is_dir():
        print(f"Error: '{directory_path}' is not a directory.")
        return
    
    # Set up input subdirectories
    input_tables_path = dir_path / "filtered_tables"
    input_metadata_path = dir_path / "filtered_metadata"
    
    if not input_tables_path.exists() or not input_metadata_path.exists():
        print(f"Error: Input directory must contain 'filtered_tables' and 'filtered_metadata' subdirectories.")
        print(f"Expected structure:")
        print(f"  {dir_path}/filtered_tables/")
        print(f"  {dir_path}/filtered_metadata/")
        return
    
    # Set up output directory
    if output_directory:
        out_path = Path(output_directory)
        output_tables_path = out_path / "capped_tables"
        output_metadata_path = out_path / "capped_metadata"
        output_tables_path.mkdir(parents=True, exist_ok=True)
        output_metadata_path.mkdir(parents=True, exist_ok=True)
        print(f"Input directory: {dir_path}")
        print(f"Output directory: {out_path}")
        print(f"Tables will be read from: {input_tables_path}")
        print(f"Metadata will be read from: {input_metadata_path}")
        print(f"Tables will be saved to: {output_tables_path}")
        print(f"Metadata will be saved to: {output_metadata_path}\n")
    else:
        output_tables_path = input_tables_path
        output_metadata_path = input_metadata_path
        print(f"Processing files in-place: {dir_path}\n")
    
    # Find all JSON files with 'github_csv' prefix in tables directory
    # Note: Table files and metadata files have the same name in separate folders
    # Table files are in tables/ folder (e.g., github_csv_36d5d569e5.json)
    # Metadata files are in metadata/ folder (e.g., github_csv_36d5d569e5.json)
    json_files = list(input_tables_path.glob('github_csv*.json'))
    
    if not json_files:
        print(f"No JSON files with 'github_csv' prefix found in '{input_tables_path}'")
        return
    
    print(f"Found {len(json_files)} files to process.\n")
    
    for file_path in json_files:
        try:
            # Read the corresponding metadata file to get row count
            # Metadata file has the SAME name as table file, just in different folder
            metadata_file_path = input_metadata_path / file_path.name
            
            if not metadata_file_path.exists():
                print(f"⚠ Skipped {file_path.name}: Corresponding metadata file not found")
                continue
            
            # Read metadata to get original row count
            with open(metadata_file_path, 'r', encoding='utf-8') as f:
                input_metadata = json.load(f)
            
            # Extract num_rows from metadata
            if 'dimensions' in input_metadata and 'rows' in input_metadata['dimensions']:
                original_count = input_metadata['dimensions']['rows']
            else:
                print(f"⚠ Skipped {file_path.name}: 'rows' not found in metadata")
                continue
            
            if original_count is None:
                print(f"⚠ Skipped {file_path.name}: 'rows' value is None in metadata")
                continue
            
            # Read the data JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract rows from the data structure
            # Expected format: {"columns": [...], "data": [[row1], [row2], ...]}
            if isinstance(data, dict) and 'data' in data:
                rows = data['data']
            elif isinstance(data, list):
                rows = data
            else:
                print(f"⚠ Skipped {file_path.name}: Expected 'data' key or array format")
                continue
            
            # Randomly select a cap between 10-20
            cap = random.randint(10, 20)
            
            # Cap the rows
            if original_count > cap:
                rows = rows[:cap]
                
                # Write back to file
                if isinstance(data, list):
                    data = rows
                else:
                    # Update the 'data' key specifically
                    data['data'] = rows
                
                # Write to output tables directory
                output_file_path = output_tables_path / file_path.name
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                
                print(f"✓ {file_path.name}")
                print(f"  Rows: {original_count} → {cap}")
                
                # Update corresponding metadata file
                update_metadata(input_metadata_path, output_metadata_path, file_path.name, cap)
            else:
                print(f"- {file_path.name} (already {original_count} rows, no capping needed)")
        
        except json.JSONDecodeError:
            print(f"✗ {file_path.name}: Invalid JSON format")
        except Exception as e:
            print(f"✗ {file_path.name}: Error - {str(e)}")
    
    print("\nProcessing complete!")


def update_metadata(input_metadata_path, output_metadata_path, filename, new_count):
    """
    Copy metadata file from input to output and update only the row count.
    
    Args:
        input_metadata_path (Path): Path to read metadata from
        output_metadata_path (Path): Path to save metadata to
        filename (str): Full filename with .json extension (e.g., 'github_csv_36d5d569e5.json')
        new_count (int): New row count after capping
    """
    
    source_metadata_path = input_metadata_path / filename
    
    try:
        # Check if metadata file exists in input
        if source_metadata_path.exists():
            with open(source_metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            # Create new metadata file if it doesn't exist
            metadata = {}
        
        # Update num_rows in both locations
        if 'dimensions' in metadata and 'rows' in metadata['dimensions']:
            metadata['dimensions']['rows'] = new_count
        
        if 'original_metadata' in metadata and 'table_statistics' in metadata['original_metadata']:
            metadata['original_metadata']['table_statistics']['num_rows'] = new_count
            # Also update num_cells
            num_cols = metadata['original_metadata']['table_statistics'].get('num_columns', 0)
            if num_cols > 0:
                metadata['original_metadata']['table_statistics']['num_cells'] = new_count * num_cols
        
        # Write metadata file to output metadata directory
        output_metadata_file = output_metadata_path / filename
        with open(output_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Updated metadata: {filename}")
    
    except json.JSONDecodeError:
        print(f"  ⚠ Warning: Metadata file exists but has invalid JSON: {filename}")
    except Exception as e:
        print(f"  ✗ Error updating metadata: {str(e)}")


if __name__ == "__main__":
    # Get input and output directories
    dir_path = 'data/filtered_data'
    out_dir = "data/github_capped"  # Set to None to overwrite original files
    
    output_dir = out_dir if out_dir else None
    cap_json_rows(dir_path, output_dir)