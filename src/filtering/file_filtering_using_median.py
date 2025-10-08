import json
import os
import shutil
from pathlib import Path

def load_character_stats(stats_file):
    """
    Load character statistics from JSON file including source-specific medians.
    
    Args:
        stats_file: Path to the character_stats.json file
        
    Returns:
        Tuple of (individual_files dict, source_medians dict)
    """
    try:
        with open(stats_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        individual_files = data.get('individual_files', {})
        
        # Extract median values for each source
        source_medians = {}
        by_source = data.get('by_source', {})
        for source, stats in by_source.items():
            median = stats.get('median_characters_per_file')
            if median is not None:
                source_medians[source] = median
        
        return individual_files, source_medians
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading stats file: {e}")
        return {}, {}

def get_data_source(filename):
    """
    Extract data source from filename (arxiv, wikisql, finqa).
    
    Args:
        filename: Filename string
        
    Returns:
        Source string ('arxiv', 'wikisql', 'finqa', or 'other')
    """
    filename_lower = filename.lower()
    if 'arxiv' in filename_lower:
        return 'arxiv'
    elif 'wikisql' in filename_lower:
        return 'wikisql'
    elif 'finqa' in filename_lower:
        return 'finqa'
    else:
        return 'other'

def extract_hash_from_filename(filename):
    """
    Extract the hash identifier from a filename.
    
    Args:
        filename: Filename string (e.g., 'arxiv_0b8207bf5f.json')
        
    Returns:
        Hash string (e.g., '0b8207bf5f')
    """
    # Remove .json extension if present
    if filename.endswith('.json'):
        filename = filename[:-5]
    
    # Assuming format is 'source_hash', extract the hash part
    parts = filename.split('_')
    if len(parts) >= 2:
        return '_'.join(parts[1:])  # Return everything after first underscore
    return filename

def find_file_by_name(directory, filename):
    """
    Find a file in directory by exact filename.
    
    Args:
        directory: Directory to search
        filename: Filename to find
        
    Returns:
        Path to file if found, None otherwise
    """
    directory = Path(directory)
    
    # Search in directory and subdirectories
    for file_path in directory.glob('**/' + filename):
        return file_path
    
    return None

def filter_and_copy_tables(stats_file, data_directory, metadata_directory, 
                           output_directory, use_source_median=True, 
                           global_min_char_count=None, dry_run=True):
    """
    Filter tables based on source-specific median character counts and copy them to output directory.
    
    Args:
        stats_file: Path to character_stats.json
        data_directory: Directory containing data JSON files
        metadata_directory: Directory containing metadata JSON files
        output_directory: Directory to copy filtered files
        use_source_median: If True, use source-specific medians; if False, use global_min_char_count
        global_min_char_count: Global minimum character count (used if use_source_median=False)
        dry_run: If True, only list files without copying them
    """
    # Load character statistics and source medians
    char_stats, source_medians = load_character_stats(stats_file)
    
    if not char_stats:
        print("No character statistics found!")
        return
    
    print(f"Loaded statistics for {len(char_stats)} files")
    
    if use_source_median:
        print(f"\nUsing source-specific median thresholds:")
        for source, median in source_medians.items():
            print(f"  {source}: {median} characters")
    else:
        print(f"\nUsing global minimum character count: {global_min_char_count}")
    print()
    
    # Filter files based on character count
    files_to_keep = []
    files_to_remove = []
    filter_summary_by_source = {source: {'kept': 0, 'removed': 0} for source in source_medians.keys()}
    filter_summary_by_source['other'] = {'kept': 0, 'removed': 0}
    
    for filename, char_count in char_stats.items():
        source = get_data_source(filename)
        
        # Determine threshold for this file
        if use_source_median:
            threshold = source_medians.get(source, 0)
        else:
            threshold = global_min_char_count if global_min_char_count is not None else 0
        
        # Apply filter
        if char_count >= threshold:
            files_to_keep.append({
                'filename': filename,
                'char_count': char_count,
                'source': source,
                'threshold': threshold
            })
            filter_summary_by_source[source]['kept'] += 1
        else:
            files_to_remove.append({
                'filename': filename,
                'char_count': char_count,
                'source': source,
                'threshold': threshold
            })
            filter_summary_by_source[source]['removed'] += 1
    
    # Sort by character count for better visualization
    files_to_keep.sort(key=lambda x: x['char_count'], reverse=True)
    files_to_remove.sort(key=lambda x: x['char_count'])
    
    print(f"✓ Files meeting criteria: {len(files_to_keep)}")
    print(f"✗ Files below threshold: {len(files_to_remove)}\n")
    
    print("Filter results by source:")
    for source, counts in filter_summary_by_source.items():
        total = counts['kept'] + counts['removed']
        if total > 0:
            percentage = (counts['kept'] / total * 100) if total > 0 else 0
            print(f"  {source}: {counts['kept']}/{total} kept ({percentage:.1f}%)")
    print()
    
    if files_to_remove:
        print(f"Files to be excluded (showing first 10):")
        for item in files_to_remove[:10]:
            print(f"  - {item['filename']} ({item['source']}): {item['char_count']} chars (threshold: {item['threshold']})")
        if len(files_to_remove) > 10:
            print(f"  ... and {len(files_to_remove) - 10} more")
        print()
    
    if not dry_run:
        # Create output directories
        output_data_dir = Path(output_directory) / "filtered_tables"
        output_metadata_dir = Path(output_directory) / "filtered_metadata"
        output_data_dir.mkdir(parents=True, exist_ok=True)
        output_metadata_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*50)
        print("COPYING FILES...")
        print("="*50)
        
        copied_count = 0
        missing_files = []
        
        for item in files_to_keep:
            filename = item['filename']
            char_count = item['char_count']
            source = item['source']
            
            # Find data file
            data_file = find_file_by_name(data_directory, filename)
            
            if data_file:
                # Copy data file
                try:
                    dest_path = output_data_dir / filename
                    shutil.copy2(data_file, dest_path)
                    print(f"✓ Copied data: {filename} ({source}, {char_count} chars)")
                    copied_count += 1
                    
                    # Find and copy corresponding metadata file
                    file_hash = extract_hash_from_filename(filename)
                    metadata_file = None
                    
                    for meta_path in Path(metadata_directory).glob('**/*.json'):
                        if file_hash in meta_path.stem:
                            metadata_file = meta_path
                            break
                    
                    if metadata_file:
                        dest_meta_path = output_metadata_dir / metadata_file.name
                        shutil.copy2(metadata_file, dest_meta_path)
                        print(f"  ✓ Copied metadata: {metadata_file.name}")
                        copied_count += 1
                    else:
                        print(f"  ⚠ Metadata not found for {filename}")
                    
                except Exception as e:
                    print(f"✗ Error copying {filename}: {e}")
            else:
                missing_files.append(filename)
        
        print(f"\n{'='*50}")
        print(f"Total files copied: {copied_count}")
        print(f"Output directory: {output_directory}")
        
        if missing_files:
            print(f"\n⚠ Warning: {len(missing_files)} files not found in data directory")
            print("Missing files (showing first 5):")
            for fname in missing_files[:5]:
                print(f"  - {fname}")
    
    else:
        print("="*50)
        print("DRY RUN MODE - No files were copied")
        print("="*50)
        print(f"Would copy {len(files_to_keep)} data files + metadata to: {output_directory}")
        print("Set dry_run=False to actually copy the files.")
    
    # Save comprehensive summary
    summary = {
        'filter_criteria': {
            'use_source_median': use_source_median,
            'source_medians': source_medians if use_source_median else None,
            'global_min_char_count': global_min_char_count if not use_source_median else None,
            'stats_file': str(stats_file),
            'data_directory': str(data_directory),
            'metadata_directory': str(metadata_directory),
            'output_directory': str(output_directory)
        },
        'statistics': {
            'total_files_analyzed': len(char_stats),
            'files_kept': len(files_to_keep),
            'files_excluded': len(files_to_remove),
            'percentage_kept': round((len(files_to_keep) / len(char_stats) * 100), 2) if char_stats else 0,
            'by_source': filter_summary_by_source
        },
        'files_kept': files_to_keep,
        'files_excluded': files_to_remove
    }
    
    # Always save summary
    summary_file = 'src/summary/filter_summary.json'
    summary_file = Path(summary_file)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Detailed filter summary saved to: {summary_file}")
    
    # Return summary for further processing if needed
    return summary

# Example usage
if __name__ == "__main__":
    # Configuration
    stats_file = "src/summary/character_stats.json"
    table_dir = "data/processed/tables"
    metadata_dir = "data/processed/metadata"
    output_dir = "data/filteredData"
    
    # Run filtering with source-specific medians (dry run first)
    print("=== DRY RUN MODE (Using Source-Specific Medians) ===\n")
    filter_and_copy_tables(
        stats_file=stats_file,
        data_directory=table_dir,
        metadata_directory=metadata_dir,
        output_directory=output_dir,
        use_source_median=True,  # Use source-specific medians
        dry_run=True
    )
    
    # Uncomment to actually copy files
    print("\n\n=== COPYING FILES (Using Source-Specific Medians) ===\n")
    filter_and_copy_tables(
        stats_file=stats_file,
        data_directory=table_dir,
        metadata_directory=metadata_dir,
        output_directory=output_dir,
        use_source_median=True,  # Use source-specific medians
        dry_run=False
    )
    
    # Alternative: Use global threshold instead
    # print("\n\n=== Using Global Threshold ===\n")
    # filter_and_copy_tables(
    #     stats_file=stats_file,
    #     data_directory=table_dir,
    #     metadata_directory=metadata_dir,
    #     output_directory=output_dir,
    #     use_source_median=False,
    #     global_min_char_count=200,
    #     dry_run=False
    # )





# import json
# import os
# import shutil
# from pathlib import Path

# def load_character_stats(stats_file):
#     """
#     Load character statistics from JSON file.
    
#     Args:
#         stats_file: Path to the character_stats.json file
        
#     Returns:
#         Dictionary of filename -> character count
#     """
#     try:
#         with open(stats_file, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         return data.get('individual_files', {})
#     except (json.JSONDecodeError, FileNotFoundError) as e:
#         print(f"Error reading stats file: {e}")
#         return {}

# def extract_hash_from_filename(filename):
#     """
#     Extract the hash identifier from a filename.
    
#     Args:
#         filename: Filename string (e.g., 'arxiv_0b8207bf5f.json')
        
#     Returns:
#         Hash string (e.g., '0b8207bf5f')
#     """
#     # Remove .json extension if present
#     if filename.endswith('.json'):
#         filename = filename[:-5]
    
#     # Assuming format is 'source_hash', extract the hash part
#     parts = filename.split('_')
#     if len(parts) >= 2:
#         return '_'.join(parts[1:])  # Return everything after first underscore
#     return filename

# def find_file_by_name(directory, filename):
#     """
#     Find a file in directory by exact filename.
    
#     Args:
#         directory: Directory to search
#         filename: Filename to find
        
#     Returns:
#         Path to file if found, None otherwise
#     """
#     directory = Path(directory)
    
#     # Search in directory and subdirectories
#     for file_path in directory.glob('**/' + filename):
#         return file_path
    
#     return None

# def filter_and_copy_tables(stats_file, data_directory, metadata_directory, 
#                            output_directory, min_char_count=150, dry_run=True):
#     """
#     Filter tables based on character count and copy them to output directory.
    
#     Args:
#         stats_file: Path to character_stats.json
#         data_directory: Directory containing data JSON files
#         metadata_directory: Directory containing metadata JSON files
#         output_directory: Directory to copy filtered files
#         min_char_count: Minimum English character count threshold
#         dry_run: If True, only list files without copying them
#     """
#     # Load character statistics
#     char_stats = load_character_stats(stats_file)
    
#     if not char_stats:
#         print("No character statistics found!")
#         return
    
#     print(f"Loaded statistics for {len(char_stats)} files")
#     print(f"Filtering with minimum character count: {min_char_count}\n")
    
#     # Filter files based on character count
#     files_to_keep = []
#     files_to_remove = []
    
#     for filename, char_count in char_stats.items():
#         if char_count >= min_char_count:
#             files_to_keep.append({
#                 'filename': filename,
#                 'char_count': char_count
#             })
#         else:
#             files_to_remove.append({
#                 'filename': filename,
#                 'char_count': char_count
#             })
    
#     # Sort by character count for better visualization
#     files_to_keep.sort(key=lambda x: x['char_count'], reverse=True)
#     files_to_remove.sort(key=lambda x: x['char_count'])
    
#     print(f"✓ Files meeting criteria (>= {min_char_count} chars): {len(files_to_keep)}")
#     print(f"✗ Files below threshold: {len(files_to_remove)}\n")
    
#     if files_to_remove:
#         print(f"Files to be excluded (showing first 10):")
#         for item in files_to_remove[:10]:
#             print(f"  - {item['filename']}: {item['char_count']} chars")
#         if len(files_to_remove) > 10:
#             print(f"  ... and {len(files_to_remove) - 10} more")
#         print()
    
#     if not dry_run:
#         # Create output directories
#         output_data_dir = Path(output_directory) / "filtered_tables"
#         output_metadata_dir = Path(output_directory) / "filtered_metadata"
#         output_data_dir.mkdir(parents=True, exist_ok=True)
#         output_metadata_dir.mkdir(parents=True, exist_ok=True)
        
#         print("="*50)
#         print("COPYING FILES...")
#         print("="*50)
        
#         copied_count = 0
#         missing_files = []
        
#         for item in files_to_keep:
#             filename = item['filename']
#             char_count = item['char_count']
            
#             # Find data file
#             data_file = find_file_by_name(data_directory, filename)
            
#             if data_file:
#                 # Copy data file
#                 try:
#                     dest_path = output_data_dir / filename
#                     shutil.copy2(data_file, dest_path)
#                     print(f"✓ Copied data: {filename} ({char_count} chars)")
#                     copied_count += 1
                    
#                     # Find and copy corresponding metadata file
#                     file_hash = extract_hash_from_filename(filename)
#                     metadata_file = None
                    
#                     for meta_path in Path(metadata_directory).glob('**/*.json'):
#                         if file_hash in meta_path.stem:
#                             metadata_file = meta_path
#                             break
                    
#                     if metadata_file:
#                         dest_meta_path = output_metadata_dir / metadata_file.name
#                         shutil.copy2(metadata_file, dest_meta_path)
#                         print(f"  ✓ Copied metadata: {metadata_file.name}")
#                         copied_count += 1
#                     else:
#                         print(f"  ⚠ Metadata not found for {filename}")
                    
#                 except Exception as e:
#                     print(f"✗ Error copying {filename}: {e}")
#             else:
#                 missing_files.append(filename)
        
#         print(f"\n{'='*50}")
#         print(f"Total files copied: {copied_count}")
#         print(f"Output directory: {output_directory}")
        
#         if missing_files:
#             print(f"\n⚠ Warning: {len(missing_files)} files not found in data directory")
#             print("Missing files (showing first 5):")
#             for fname in missing_files[:5]:
#                 print(f"  - {fname}")
    
#     else:
#         print("="*50)
#         print("DRY RUN MODE - No files were copied")
#         print("="*50)
#         print(f"Would copy {len(files_to_keep)} data files + metadata to: {output_directory}")
#         print("Set dry_run=False to actually copy the files.")
    
#     # Save comprehensive summary
#     summary = {
#         'filter_criteria': {
#             'min_char_count': min_char_count,
#             'stats_file': str(stats_file),
#             'data_directory': str(data_directory),
#             'metadata_directory': str(metadata_directory),
#             'output_directory': str(output_directory)
#         },
#         'statistics': {
#             'total_files_analyzed': len(char_stats),
#             'files_kept': len(files_to_keep),
#             'files_excluded': len(files_to_remove),
#             'percentage_kept': round((len(files_to_keep) / len(char_stats) * 100), 2) if char_stats else 0
#         },
#         'files_kept': files_to_keep,
#         'files_excluded': files_to_remove
#     }
    
#     # Always save summary
#     summary_file = Path(output_directory) / 'filter_summary.json' if not dry_run else 'filter_summary.json'
#     summary_file = Path(summary_file)
#     summary_file.parent.mkdir(parents=True, exist_ok=True)
    
#     with open(summary_file, 'w', encoding='utf-8') as f:
#         json.dump(summary, f, indent=2)
#     print(f"\n✓ Detailed filter summary saved to: {summary_file}")
    
#     # Return summary for further processing if needed
#     return summary

# # Example usage
# if __name__ == "__main__":
#     # Configuration
#     stats_file = "src/filtering/character_stats.json"
#     table_dir = "data/processed/tables"
#     metadata_dir = "data/processed/metadata"
#     output_dir = "src/filtering/filteredData"
#     min_chars = 200  # Minimum character threshold
    
#     # Run filtering (dry run first)
#     print("=== DRY RUN MODE ===\n")
#     filter_and_copy_tables(
#         stats_file=stats_file,
#         data_directory=table_dir,
#         metadata_directory=metadata_dir,
#         output_directory=output_dir,
#         min_char_count=min_chars,
#         dry_run=True
#     )
    
#     # Uncomment to actually copy files
#     print("\n\n=== COPYING FILES ===\n")
#     filter_and_copy_tables(
#         stats_file=stats_file,
#         data_directory=table_dir,
#         metadata_directory=metadata_dir,
#         output_directory=output_dir,
#         min_char_count=min_chars,
#         dry_run=False
#     )