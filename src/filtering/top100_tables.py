import json
import os
from pathlib import Path
from typing import List, Dict
import shutil

def load_scores_from_json(scores_file: str) -> Dict[str, int]:
    """Load scores from the JSON file produced by tables_scoring.py."""
    with open(scores_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract filename -> score mapping
    scores = {}
    if 'tables' in data:
        for table in data['tables']:
            filename = table['file']
            score = table['score']
            scores[filename] = score
    
    return scores

def filter_top_tables_by_score(
    tables_dir: str,
    scores_dir: str,
    output_dir: str ,
    reports_dir: str,
    metadata_dir: str = None,
    metadata_output_dir: str = None,
    tables_per_category: int = 100,
    categories: List[str] = None
) -> Dict[str, List[Dict]]:
    """
    Filter top N tables per category based on semantic preservation scores.
    
    Args:
        tables_dir: Directory containing table JSON files
        scores_dir: Directory containing score JSON files (e.g., arxiv_table_scores.json)
        output_dir: Output directory for selected tables
        reports_dir: Output directory for selection reports
        metadata_dir: Directory containing metadata files (optional)
        metadata_output_dir: Output directory for metadata files. If None, defaults to {output_dir}/metadata
        tables_per_category: Number of top tables to select per category
        categories: List of specific categories to process (e.g., ['arxiv', 'finqa']). 
                   If None, processes all available categories.
    """
    
    tables_path = Path(tables_dir)
    scores_path = Path(scores_dir)
    output_path = Path(output_dir)
    reports_path = Path(reports_dir)
    metadata_path = Path(metadata_dir) if metadata_dir else None
    
    # Set metadata output directory
    if metadata_dir:
        if metadata_output_dir:
            metadata_output_path = Path(metadata_output_dir)
        else:
            metadata_output_path = output_path / 'metadata'
    else:
        metadata_output_path = None
    
    print("="*70)
    print("TABLE FILTERING BY SEMANTIC PRESERVATION SCORE")
    print("="*70)
    
    # Find all score files
    score_files = list(scores_path.glob('*_table_scores.json'))
    
    if not score_files:
        print(f"\n✗ Error: No score files found in {scores_dir}")
        print("   Expected files like: arxiv_table_scores.json, finqa_table_scores.json")
        return {}
    
    # Filter by specific categories if provided
    if categories:
        categories_lower = [c.lower() for c in categories]
        score_files = [
            sf for sf in score_files 
            if sf.stem.replace('_table_scores', '') in categories_lower
        ]
        
        if not score_files:
            print(f"\n✗ Error: No score files found for categories: {categories}")
            return {}
        
        print(f"\n🎯 Processing specific categories: {', '.join(categories)}")
    else:
        print(f"\n🎯 Processing all available categories")
    
    print(f"\n📊 Found {len(score_files)} category score files:")
    for sf in score_files:
        print(f"   • {sf.name}")
    
    if metadata_dir:
        print(f"\n📂 Metadata directory: {metadata_dir}")
    
    # Process each category
    all_selected = {}
    
    for score_file in score_files:
        # Extract category name (e.g., 'arxiv' from 'arxiv_table_scores.json')
        category = score_file.stem.replace('_table_scores', '')
        
        print(f"\n{'='*70}")
        print(f"Processing {category.upper()} category")
        print(f"{'='*70}")
        
        # Load scores
        try:
            scores = load_scores_from_json(str(score_file))
            print(f"   ✓ Loaded scores for {len(scores)} tables")
        except Exception as e:
            print(f"   ✗ Error loading {score_file.name}: {e}")
            continue
        
        if not scores:
            print(f"   ⚠️  No scores found in {score_file.name}")
            continue
        
        # Sort tables by score (highest first)
        sorted_tables = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top N
        n_select = min(tables_per_category, len(sorted_tables))
        top_tables = sorted_tables[:n_select]
        
        print(f"   📈 Score range: {top_tables[-1][1]} to {top_tables[0][1]}")
        print(f"   🎯 Selecting top {n_select} tables")
        
        # Check if tables exist
        selected_tables = []
        missing_count = 0
        missing_metadata_count = 0
        
        for filename, score in top_tables:
            table_file = tables_path / filename
            
            if not table_file.exists():
                missing_count += 1
                continue
            
            # Check for metadata file if metadata_dir is provided
            metadata_file = None
            if metadata_path:
                metadata_file = metadata_path / filename
                if not metadata_file.exists():
                    missing_metadata_count += 1
                    metadata_file = None
            
            # Load table to get dimensions
            try:
                with open(table_file, 'r', encoding='utf-8') as f:
                    table_data = json.load(f)
                
                num_rows = len(table_data.get('data', []))
                num_cols = len(table_data.get('columns', []))
                
                selected_tables.append({
                    'filename': filename,
                    'score': score,
                    'table_path': str(table_file),
                    'metadata_path': str(metadata_file) if metadata_file else None,
                    'num_rows': num_rows,
                    'num_columns': num_cols,
                    'total_cells': num_rows * num_cols
                })
            except Exception as e:
                print(f"   ⚠️  Error reading {filename}: {e}")
                continue
        
        if missing_count > 0:
            print(f"   ⚠️  Warning: {missing_count} table files not found")
        
        if missing_metadata_count > 0 and metadata_path:
            print(f"   ⚠️  Warning: {missing_metadata_count} metadata files not found")
        
        all_selected[category] = selected_tables
        
        # Print category statistics
        if selected_tables:
            avg_score = sum(t['score'] for t in selected_tables) / len(selected_tables)
            avg_rows = sum(t['num_rows'] for t in selected_tables) / len(selected_tables)
            avg_cols = sum(t['num_columns'] for t in selected_tables) / len(selected_tables)
            
            print(f"\n   ✓ Selected {len(selected_tables)} tables:")
            print(f"      Avg Score: {avg_score:.2f}/10")
            print(f"      Avg Size: {avg_rows:.0f} rows × {avg_cols:.0f} cols")
    
    # Create single output directory and copy all files
    print(f"\n{'='*70}")
    print("COPYING FILES")
    print(f"{'='*70}")
    
    # Create single directory for all tables
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create metadata output directory if needed
    if metadata_output_path:
        metadata_output_path.mkdir(parents=True, exist_ok=True)
    
    total_copied = 0
    total_metadata_copied = 0
    
    for category, tables in all_selected.items():
        if not tables:
            continue
        
        print(f"\n🗂️  {category}: Copying {len(tables)} tables...")
        
        # Copy each table to the main output directory
        for table_info in tables:
            src_path = Path(table_info['table_path'])
            dest_path = output_path / table_info['filename']
            
            try:
                shutil.copy2(src_path, dest_path)
                total_copied += 1
            except Exception as e:
                print(f"   ✗ Error copying {table_info['filename']}: {e}")
        
        # Copy metadata files if they exist
        if metadata_output_path:
            for table_info in tables:
                if table_info['metadata_path']:
                    src_meta_path = Path(table_info['metadata_path'])
                    if src_meta_path.exists():
                        dest_meta_path = metadata_output_path / table_info['filename']
                        try:
                            shutil.copy2(src_meta_path, dest_meta_path)
                            total_metadata_copied += 1
                        except Exception as e:
                            print(f"   ✗ Error copying metadata {table_info['filename']}: {e}")
        
        print(f"   ✓ Copied {len(tables)} files")
        if metadata_output_path:
            print(f"   ✓ Copied {total_metadata_copied} metadata files")
    
    # Generate comprehensive report
    print(f"\n{'='*70}")
    print("GENERATING REPORTS")
    print(f"{'='*70}")
    
    # Create reports directory
    reports_path.mkdir(parents=True, exist_ok=True)
    print(f"\n📊 Reports directory: {reports_path}")
    
    report = {
        'selection_criteria': 'semantic_preservation_score',
        'tables_per_category_target': tables_per_category,
        'total_tables_selected': total_copied,
        'total_metadata_files_copied': total_metadata_copied,
        'metadata_included': metadata_path is not None,
        'categories': {}
    }
    
    for category, tables in all_selected.items():
        if not tables:
            continue
        
        # Calculate statistics
        scores = [t['score'] for t in tables]
        
        report['categories'][category] = {
            'count': len(tables),
            'score_stats': {
                'min': min(scores),
                'max': max(scores),
                'avg': sum(scores) / len(scores),
                'median': sorted(scores)[len(scores)//2]
            },
            'avg_rows': sum(t['num_rows'] for t in tables) / len(tables),
            'avg_columns': sum(t['num_columns'] for t in tables) / len(tables),
            'avg_cells': sum(t['total_cells'] for t in tables) / len(tables),
            'score_distribution': {
                'score_10': sum(1 for t in tables if t['score'] == 10),
                'score_9': sum(1 for t in tables if t['score'] == 9),
                'score_8': sum(1 for t in tables if t['score'] == 8),
                'score_7': sum(1 for t in tables if t['score'] == 7),
                'score_6': sum(1 for t in tables if t['score'] == 6),
                'score_5_or_less': sum(1 for t in tables if t['score'] <= 5)
            },
            'top_10_tables': [
                {
                    'filename': t['filename'],
                    'score': t['score'],
                    'rows': t['num_rows'],
                    'columns': t['num_columns'],
                    'has_metadata': t['metadata_path'] is not None
                }
                for t in tables[:10]
            ],
            'all_tables': tables
        }
        
        # Save category-specific report in reports directory
        cat_report_path = reports_path / f'{category}_selection_report.json'
        with open(cat_report_path, 'w', encoding='utf-8') as f:
            json.dump(report['categories'][category], f, indent=2)
        
        print(f"   ✓ Saved {category} report: {cat_report_path}")
    
    # Save master report in reports directory
    master_report_path = reports_path / 'master_selection_report.json'
    with open(master_report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"   ✓ Saved master report: {master_report_path}")
    
    # Create summary CSV in reports directory
    import csv
    summary_csv_path = reports_path / 'selection_summary.csv'
    with open(summary_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Category', 'Tables', 'Min Score', 'Max Score', 'Avg Score', 
                        'Avg Rows', 'Avg Cols', 'Score 10', 'Score 9', 'Score 8', 'Metadata Files'])
        
        for category in sorted(all_selected.keys()):
            if category not in report['categories']:
                continue
            cat_data = report['categories'][category]
            dist = cat_data['score_distribution']
            metadata_count = sum(1 for t in all_selected[category] if t['metadata_path'] is not None)
            writer.writerow([
                category,
                cat_data['count'],
                cat_data['score_stats']['min'],
                cat_data['score_stats']['max'],
                f"{cat_data['score_stats']['avg']:.2f}",
                f"{cat_data['avg_rows']:.0f}",
                f"{cat_data['avg_columns']:.0f}",
                dist['score_10'],
                dist['score_9'],
                dist['score_8'],
                metadata_count
            ])
    
    print(f"   ✓ Saved summary CSV: {summary_csv_path}")
    
    # Print final summary
    print(f"\n{'='*70}")
    print("✅ SELECTION COMPLETE")
    print(f"{'='*70}")
    print(f"\n📊 Summary:")
    print(f"   Total tables selected: {total_copied}")
    print(f"   Total metadata files copied: {total_metadata_copied}")
    print(f"   Categories: {len(all_selected)}")
    print(f"\n📂 Tables directory: {output_path}")
    if metadata_output_path:
        print(f"📂 Metadata directory: {metadata_output_path}")
    print(f"📄 Reports directory: {reports_path}")
    
    for category in sorted(all_selected.keys()):
        if category not in report['categories']:
            continue
        tables = all_selected[category]
        cat_data = report['categories'][category]
        stats = cat_data['score_stats']
        
        print(f"\n📦 {category.upper()} ({len(tables)} tables):")
        print(f"   Score Range: {stats['min']}-{stats['max']}/10 (avg: {stats['avg']:.2f})")
        print(f"   Avg Size: {cat_data['avg_rows']:.0f} rows × {cat_data['avg_columns']:.0f} cols")
        print(f"   Score Distribution: 10☆={cat_data['score_distribution']['score_10']}, "
              f"9☆={cat_data['score_distribution']['score_9']}, "
              f"8☆={cat_data['score_distribution']['score_8']}")
    
    print(f"\n📄 Reports:")
    print(f"   Master: {master_report_path}")
    print(f"   Summary: {summary_csv_path}")
    
    return all_selected


if __name__ == "__main__":
    import sys
    
    # Configuration
    TABLES_DIR = "data/github_capped/capped_tables"
    SCORES_DIR = "src/summary/scoring"
    OUTPUT_DIR = "data/benchmark_dataset/benchmark_tables"
    REPORTS_DIR = "src/summary/top100_tables"
    METADATA_DIR = "data/github_capped/capped_metadata"  # Add your metadata directory
    METADATA_DIR_OUTPUT = "data/benchmark_dataset/benchmark_metadata"  # Output directory for metadata files
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        # Individual category mode: python top100_tables.py arxiv finqa
        categories_to_process = sys.argv[1:]
        print(f"\n🎯 Individual Category Mode")
        print(f"   Processing: {', '.join(categories_to_process)}\n")
        
        selected = filter_top_tables_by_score(
            tables_dir=TABLES_DIR,
            scores_dir=SCORES_DIR,
            output_dir=OUTPUT_DIR,
            reports_dir=REPORTS_DIR,
            metadata_dir=METADATA_DIR,
            metadata_output_dir=METADATA_DIR_OUTPUT,  # Specify custom metadata output directory
            tables_per_category=100,
            categories=categories_to_process
        )
    else:
        # All categories mode
        print("\n🎯 All Categories Mode")
        print("   Processing all available categories\n")
        print("   💡 Tip: To process specific categories, use:")
        print("      python top100_tables.py arxiv finqa\n")
        
        selected = filter_top_tables_by_score(
            tables_dir=TABLES_DIR,
            scores_dir=SCORES_DIR,
            output_dir=OUTPUT_DIR,
            reports_dir=REPORTS_DIR,
            metadata_dir=METADATA_DIR,
            metadata_output_dir=METADATA_DIR_OUTPUT,  # Specify custom metadata output directory
            tables_per_category=100,
            categories=None  # Process all
        )
    
    print("\n✅ Top scored tables ready!")
    print(f"\nℹ️  Tables are selected from score files in: {SCORES_DIR}")
    print(f"📂 Tables saved to: {OUTPUT_DIR}")
    print(f"📂 Metadata saved to: {METADATA_DIR_OUTPUT if METADATA_DIR_OUTPUT else 'Not configured'}")
    print(f"📄 Reports saved to: {REPORTS_DIR}")
    
    if len(sys.argv) <= 1:
        print("\n💡 Usage examples:")
        print("   python top100_tables.py              # Process all categories")
        print("   python top100_tables.py arxiv        # Process only arxiv")
        print("   python top100_tables.py arxiv finqa  # Process arxiv and finqa")