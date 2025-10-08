import json
import os
import re
from pathlib import Path
from typing import List, Dict
import shutil

def calculate_english_content_ratio(data) -> float:
    """Calculate ratio of English content in table data."""
    text_content = json.dumps(data)
    words = re.findall(r'\b[a-zA-Z]+\b', text_content)
    total_chars = len(text_content)
    english_chars = sum(len(w) for w in words)
    return english_chars / max(total_chars, 1)

def assess_qa_quality(table_data: Dict, metadata: Dict) -> Dict:
    """Assess how suitable a table is for QA training."""
    quality_score = 0
    issues = []
    
    # Get table structure
    columns = table_data.get('columns', [])
    data_rows = table_data.get('data', [])
    
    if not columns or not data_rows:
        return {'score': 0, 'issues': ['No columns or data found']}
    
    num_cols = len(columns)
    num_rows = len(data_rows)
    
    # 1. English content quality (0-30 points)
    english_ratio = calculate_english_content_ratio(table_data)
    english_score = english_ratio * 30
    quality_score += english_score
    
    if english_ratio < 0.5:
        issues.append(f'Low English content: {english_ratio:.2%}')
    
    # 2. Size and information density (0-25 points)
    total_cells = num_rows * num_cols
    if total_cells >= 100:
        quality_score += 25
    elif total_cells >= 50:
        quality_score += 15
    elif total_cells >= 20:
        quality_score += 10
    else:
        quality_score += 5
        issues.append(f'Small table: {total_cells} cells')
    
    # 3. Column diversity (0-15 points)
    if num_cols >= 5:
        quality_score += 15
    elif num_cols >= 3:
        quality_score += 10
    else:
        quality_score += 5
        issues.append(f'Few columns: {num_cols}')
    
    # 4. Row count (0-15 points)
    if num_rows >= 20:
        quality_score += 15
    elif num_rows >= 10:
        quality_score += 10
    elif num_rows >= 5:
        quality_score += 5
    else:
        issues.append(f'Few rows: {num_rows}')
    
    # 5. Content richness (0-15 points)
    # Check average cell content length
    cell_contents = []
    for row in data_rows:
        for cell in row:
            if cell:
                cell_contents.append(str(cell))
    
    if cell_contents:
        avg_cell_len = sum(len(c) for c in cell_contents) / len(cell_contents)
        if avg_cell_len >= 20:  # Rich content
            quality_score += 15
        elif avg_cell_len >= 10:
            quality_score += 10
        elif avg_cell_len >= 5:
            quality_score += 5
        else:
            issues.append(f'Sparse content: avg {avg_cell_len:.1f} chars/cell')
    
    return {
        'score': quality_score,
        'english_ratio': english_ratio,
        'issues': issues
    }

def get_source_prefix(filename: str) -> str:
    """Extract source prefix from filename (e.g., 'arxiv', 'finqa', 'wikisql')."""
    # Split by underscore and get the first part
    parts = filename.split('_')
    if parts:
        return parts[0].lower()
    return 'unknown'

def filter_tables_for_qa_training(
    tables_dir: str,
    metadata_dir: str,
    output_dir: str = 'qa_training_tables',
    tables_per_category: int = 100,
    min_quality_score: float = 40.0
) -> Dict[str, List[Dict]]:
    """Filter tables optimized for QA training, categorized by source prefix."""
    
    tables_path = Path(tables_dir)
    metadata_path = Path(metadata_dir)
    output_path = Path(output_dir)
    
    print("="*70)
    print("TABLE FILTERING FOR QA TRAINING (BY SOURCE)")
    print("="*70)
    
    # Collect all table files
    table_files = list(tables_path.glob('*.json'))
    print(f"\n📁 Found {len(table_files)} table files")
    
    # Process all tables and categorize by source prefix
    categorized_tables = {}
    
    print("\n⚙️  Processing tables...")
    for i, table_file in enumerate(table_files):
        if (i + 1) % 100 == 0:
            print(f"   Progress: {i + 1}/{len(table_files)}")
        
        metadata_file = metadata_path / table_file.name
        if not metadata_file.exists():
            continue
        
        try:
            # Load data
            with open(table_file, 'r', encoding='utf-8') as f:
                table_data = json.load(f)
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Assess QA quality
            qa_assessment = assess_qa_quality(table_data, metadata)
            
            # Skip low quality tables
            if qa_assessment['score'] < min_quality_score:
                continue
            
            # Get category from filename prefix
            category = get_source_prefix(table_file.name)
            
            # Initialize category if not exists
            if category not in categorized_tables:
                categorized_tables[category] = []
            
            # Store with metrics
            dimensions = metadata.get('dimensions', {})
            categorized_tables[category].append({
                'filename': table_file.name,
                'table_path': str(table_file),
                'metadata_path': str(metadata_file),
                'category': category,
                'qa_score': qa_assessment['score'],
                'english_ratio': qa_assessment['english_ratio'],
                'num_rows': dimensions.get('rows', 0),
                'num_columns': dimensions.get('columns', 0),
                'char_count': len(json.dumps(table_data)),
                'issues': qa_assessment['issues']
            })
        
        except Exception as e:
            print(f"   ⚠️  Error processing {table_file.name}: {e}")
            continue
    
    # Sort each category by QA score
    for category in categorized_tables:
        categorized_tables[category].sort(key=lambda x: x['qa_score'], reverse=True)
    
    # Print category statistics
    print(f"\n📊 Categories Found:")
    for category in sorted(categorized_tables.keys()):
        tables = categorized_tables[category]
        print(f"   {category}: {len(tables)} quality tables")
    
    # Select top N from each category
    selected_tables = {}
    total_available = 0
    total_selected = 0
    
    for category, tables in categorized_tables.items():
        n_available = len(tables)
        total_available += n_available
        n_select = min(tables_per_category, n_available)
        selected_tables[category] = tables[:n_select]
        total_selected += n_select
        
        if n_available < tables_per_category:
            print(f"\n⚠️  Warning: Only {n_available} quality tables found in '{category}' category")
            print(f"    Selecting all {n_available} tables instead of {tables_per_category}")
    
    # Create output structure
    print(f"\n📁 Creating output directories...")
    for category in selected_tables:
        cat_tables = output_path / category / 'tables'
        cat_metadata = output_path / category / 'metadata'
        cat_tables.mkdir(parents=True, exist_ok=True)
        cat_metadata.mkdir(parents=True, exist_ok=True)
    
    # Copy selected tables
    print(f"\n📋 Copying selected tables...")
    for category, tables in selected_tables.items():
        print(f"\n   {category}: {len(tables)} tables")
        for table_info in tables:
            # Copy files
            dest_table = output_path / category / 'tables' / table_info['filename']
            dest_meta = output_path / category / 'metadata' / table_info['filename']
            shutil.copy2(table_info['table_path'], dest_table)
            shutil.copy2(table_info['metadata_path'], dest_meta)
    
    # Generate comprehensive report
    report = {
        'total_tables_processed': len(table_files),
        'total_quality_tables': total_available,
        'total_tables_selected': total_selected,
        'min_quality_threshold': min_quality_score,
        'tables_per_category_target': tables_per_category,
        'categories': {}
    }
    
    for category, tables in selected_tables.items():
        if not tables:
            continue
        
        report['categories'][category] = {
            'count': len(tables),
            'avg_qa_score': sum(t['qa_score'] for t in tables) / len(tables),
            'avg_english_ratio': sum(t['english_ratio'] for t in tables) / len(tables),
            'avg_rows': sum(t['num_rows'] for t in tables) / len(tables),
            'avg_columns': sum(t['num_columns'] for t in tables) / len(tables),
            'avg_char_count': sum(t['char_count'] for t in tables) / len(tables),
            'top_10_tables': [
                {
                    'filename': t['filename'],
                    'qa_score': t['qa_score'],
                    'rows': t['num_rows'],
                    'columns': t['num_columns'],
                    'english_ratio': t['english_ratio']
                }
                for t in tables[:10]
            ],
            'all_tables': tables
        }
        
        # Save category-specific report
        cat_report_path = output_path / category / f'{category}_report.json'
        with open(cat_report_path, 'w', encoding='utf-8') as f:
            json.dump(report['categories'][category], f, indent=2)
    
    # Save master report
    master_report_path = output_path / 'master_report.json'
    with open(master_report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    # Create summary CSV
    import csv
    summary_csv_path = output_path / 'summary.csv'
    with open(summary_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Category', 'Tables Selected', 'Avg QA Score', 'Avg English %', 
                        'Avg Rows', 'Avg Cols', 'Avg Chars'])
        
        for category in sorted(selected_tables.keys()):
            cat_data = report['categories'][category]
            writer.writerow([
                category,
                cat_data['count'],
                f"{cat_data['avg_qa_score']:.1f}",
                f"{cat_data['avg_english_ratio']:.1%}",
                f"{cat_data['avg_rows']:.1f}",
                f"{cat_data['avg_columns']:.1f}",
                f"{cat_data['avg_char_count']:.0f}"
            ])
    
    # Print summary
    print("\n" + "="*70)
    print("✅ FILTERING COMPLETE")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   Total tables processed: {len(table_files)}")
    print(f"   Quality tables found: {total_available}")
    print(f"   Tables selected: {total_selected}")
    print(f"\n📁 Output directory: {output_path}")
    
    for category in sorted(selected_tables.keys()):
        tables = selected_tables[category]
        if not tables:
            continue
        cat_data = report['categories'][category]
        print(f"\n📦 {category.upper()} ({len(tables)} tables):")
        print(f"   Avg QA Score: {cat_data['avg_qa_score']:.1f}/100")
        print(f"   Avg English: {cat_data['avg_english_ratio']:.1%}")
        print(f"   Avg Size: {cat_data['avg_rows']:.0f} rows × {cat_data['avg_columns']:.0f} cols")
        print(f"   Avg Chars: {cat_data['avg_char_count']:,.0f}")
    
    print(f"\n📄 Reports saved:")
    print(f"   Master: {master_report_path}")
    print(f"   Summary CSV: {summary_csv_path}")
    for category in selected_tables.keys():
        print(f"   {category}: {output_path / category / f'{category}_report.json'}")
    
    return selected_tables

# Example usage
if __name__ == "__main__":
    # Update these paths to your directories
    TABLES_DIR = "data/filtered_data/filtered_tables"
    METADATA_DIR = "data/filtered_data/filtered_metadata"
    OUTPUT_DIR = "data/benchmark_data"
    
    # Filter 100 tables per category (based on filename prefix)
    selected = filter_tables_for_qa_training(
        tables_dir=TABLES_DIR,
        metadata_dir=METADATA_DIR,
        output_dir=OUTPUT_DIR,
        tables_per_category=100,
        min_quality_score=40.0  # Adjust this threshold if needed
    )
    
    print("\n✅ Dataset ready for QA training!")
    print("\nCategories are based on filename prefixes:")
    print("  - arxiv_*.json → arxiv category")
    print("  - finqa_*.json → finqa category")
    print("  - wikisql_*.json → wikisql category")
    print("  - etc.")