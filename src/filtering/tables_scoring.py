import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set
import re
import time
from dotenv import load_dotenv

load_dotenv()

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  Warning: google-generativeai not installed. Run: pip install google-generativeai")


class SemanticTableQualityScorer:
    """
    Scores table quality based on semantic preservation across language translation.
    Uses Gemini API to dynamically classify terms as translatable or untranslatable.
    """
    
    # Fallback terms if API is unavailable
    FALLBACK_TRANSLATABLE_TERMS = {
        'portfolio', 'strategy', 'return', 'risk', 'price', 'value',
        'investment', 'asset', 'performance', 'ratio', 'metric', 'comparison',
        'equal', 'contribution', 'distance', 'similarity', 'probability',
        'distribution', 'optimization', 'minimization', 'maximization',
        'cost', 'loss', 'gain', 'change', 'increase', 'decrease',
        'model', 'prediction', 'forecast', 'trend', 'pattern', 'cluster',
        'classification', 'regression', 'analysis', 'method', 'algorithm',
        'accuracy', 'precision', 'recall', 'f1', 'score', 'rate', 'time',
        'dataset', 'evaluation', 'benchmark', 'performance', 'result',
        'parameter', 'coefficient', 'variable', 'feature', 'input', 'output',
        'mean', 'standard', 'deviation', 'average', 'median', 'variance',
        'correlation', 'relationship', 'comparison', 'difference', 'similarity',
        'weapon', 'shotgun', 'smg', 'knife', 'handgun', 'sniper', 'sword',
        'grenade', 'launcher', 'bazooka', 'rifle', 'automatic', 'drug',
        'dataset', 'val', 'defense', 'training', 'krav', 'magma', 'billete'
    }
    
    FALLBACK_UNTRANSLATABLE_TERMS = {
        'logistic regression', 'neural network', 'support vector', 'decision tree',
        'activation function', 'neuron', 'layer', 'kernel',
        'backpropagation', 'gradient descent', 'epoch', 'batch', 'overfitting',
        'pretrained llm', 'bloom', 'gpt', 'llm', 'token', 'finetuning',
        'pretraining', 'embedding', 'attention', 'transformer', 'lstm', 'rnn',
        'cnn', 'gan', 'vae', 'bert', 'roberta', 'xlnet', 'electra',
        'autoencoder', 'boltzmann', 'hopfield', 'perceptron', 'convolution'
    }
    
    UNIVERSAL_ELEMENTS = {
        'numbers': r'\d+\.?\d*',
        'math_operators': ['+', '-', '*', '/', '=', '<', '>', '≤', '≥', '^'],
        'symbols': ['α', 'β', 'γ', 'δ', 'σ', 'λ', 'θ', 'μ', 'Δ', 'Σ', '%', '·'],
        'math_functions': ['sin', 'cos', 'tan', 'log', 'exp', 'sqrt', 'min', 'max']
    }
    
    def __init__(self, gemini_api_key: str = None, use_gemini: bool = True, cache_file: str = "term_cache.json"):
        self.use_gemini = use_gemini and GEMINI_AVAILABLE and gemini_api_key
        self.cache_file = cache_file
        self._load_cache()
        
        # Initialize Gemini if available
        if self.use_gemini:
            genai.configure(api_key=gemini_api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
            print("✓ Gemini API initialized")
        else:
            self.model = None
            if use_gemini and not GEMINI_AVAILABLE:
                print("⚠️  Gemini API requested but library not available. Using fallback terms.")
            elif use_gemini and not gemini_api_key:
                print("⚠️  Gemini API key not provided. Using fallback terms.")
        
        # Dynamic term classification
        self.translatable_terms = set(self.FALLBACK_TRANSLATABLE_TERMS)
        self.untranslatable_terms = set(self.FALLBACK_UNTRANSLATABLE_TERMS)
        
        # Tracking
        self.found_translatable_terms = {}
        self.found_untranslatable_terms = {}
        self.unknown_terms = {}
        self.gemini_classifications = {}
        self.terms_to_classify = set()

    def _load_cache(self):
        """Load previously classified terms from cache."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                cache = json.load(f)
                self.translatable_terms.update(cache.get('translatable', []))
                self.untranslatable_terms.update(cache.get('untranslatable', []))
                self.gemini_classifications.update(cache.get('all_classifications', {}))
            print(f"✓ Loaded {len(self.gemini_classifications)} cached classifications")

    def _save_cache(self):
        """Save classifications to cache."""
        cache = {
            'translatable': list(self.translatable_terms),
            'untranslatable': list(self.untranslatable_terms),
            'all_classifications': self.gemini_classifications
        }
        with open(self.cache_file, 'w') as f:
            json.dump(cache, f, indent=2)
        print(f"✓ Saved cache with {len(self.gemini_classifications)} classifications")
    
    def _is_numeric_cell(self, cell: Any) -> bool:
        """Check if cell is purely numeric (0, 1, binary labels, etc.)"""
        if isinstance(cell, (int, float)):
            return True
        if isinstance(cell, str):
            return cell.isdigit() or cell in ['0', '1', '0.0', '1.0']
        return False
    
    def _is_file_path(self, text: str) -> bool:
        """Check if text is a file path."""
        if not isinstance(text, str):
            return False
        return ('/' in text or '\\' in text or 
                text.endswith(('.json', '.csv', '.xlsx', '.jpeg', '.jpg', '.png', '.gif', '.txt')))
    
    def extract_terms_from_tables(self, directory: str, pattern: str = "*.json", max_tables: int = None) -> Set[str]:
        """Extract all unique terms from tables before processing."""
        all_terms = set()
        json_files = list(Path(directory).glob(pattern))
        
        if max_tables is not None:
            json_files = json_files[:max_tables]
        
        print(f"  Extracting terms from {len(json_files)} tables...")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    table_data = json.load(f)
                
                if "data" in table_data:
                    for row in table_data["data"]:
                        for cell in row:
                            # Skip numeric and file path cells
                            if self._is_numeric_cell(cell) or self._is_file_path(str(cell)):
                                continue
                            
                            if isinstance(cell, str):
                                words = re.findall(r'\b[a-z_]+\b', cell.lower())
                                all_terms.update(words)
                                
                                cell_lower = cell.lower()
                                phrases = re.findall(r'\b[a-z]+\s+[a-z]+(?:\s+[a-z]+)?\b', cell_lower)
                                all_terms.update(phrases)
            except Exception as e:
                print(f"  Error extracting from {json_file.name}: {e}")
        
        filtered_terms = {
            term for term in all_terms 
            if len(term) > 2 and term not in [
                'the', 'a', 'an', 'and', 'or', 'is', 'are', 'was', 'were',
                'in', 'of', 'to', 'for', 'with', 'by', 'at', 'from', 'as',
                'on', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that'
            ]
        }
        
        print(f"  ✓ Extracted {len(filtered_terms)} unique terms")
        return filtered_terms
    
    def classify_terms_with_gemini(self, terms: Set[str], batch_size: int = 1000, 
                               max_retries: int = 7, initial_delay: float = 0.5) -> Dict[str, str]:
        """Use Gemini API to classify terms with robust retry logic for rate limits."""
        if not self.use_gemini:
            return {}
        
        classifications = {}
        terms_list = list(terms)
        total_batches = (len(terms_list) + batch_size - 1) // batch_size
        
        print(f"  Classifying {len(terms_list)} terms using Gemini API ({total_batches} batches)...")
        print(f"  Max retries per batch: {max_retries}, Initial delay: {initial_delay}s")
        
        for i in range(0, len(terms_list), batch_size):
            batch = terms_list[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            batch_success = False
            
            for attempt in range(max_retries):
                try:
                    prompt = f"""Classify each term for cross-language semantic preservation.

TRANSLATABLE = Universal concepts with clear, meaningful translations in other languages
Examples: 
  - General concepts: "weather", "climate", "disaster", "environment", "temperature", "policy", "insurance"
  - Common tech terms: "data", "index", "table", "database", "timeseries", "attributes", "variable"
  - Business terms: "portfolio", "strategy", "return", "risk", "accuracy", "model", "service"
  - Legal/privacy: "privacy", "information", "user", "account", "consent", "legal"
  - Common nouns: "weapon", "knife", "shotgun", "handgun", "sword", "training", "defense"

UNTRANSLATABLE = Specialized ML/AI jargon or technical model names that lose precise meaning
Examples: 
  - ML/AI terms: "LSTM", "backpropagation", "GPT-4", "BERT", "transformer", "neural network", "gradient descent"
  - Very technical: "logistic regression", "convolutional", "recurrent neural network"

IMPORTANT GUIDELINES:
1. Common technical/domain terms ARE translatable: "weather", "climate", "disaster", "emissions", "database", "index"
2. Business and legal terms ARE translatable: "service", "policy", "insurance", "privacy", "information"
3. Common nouns for objects ARE translatable: "weapon", "knife", "handgun", "training"
4. Only mark as untranslatable if it's specialized ML/AI terminology or loses technical precision in translation
5. Acronyms for organizations (FEMA, NOAA) can be translatable - they're proper nouns but meaning is preserved

Terms to classify:
{json.dumps(batch, indent=2)}

Respond ONLY with a JSON object mapping each term to either "translatable" or "untranslatable". Example:
{{
  "weather": "translatable",
  "lstm": "untranslatable",
  "disaster": "translatable",
  "backpropagation": "untranslatable",
  "insurance": "translatable"
}}

JSON response:"""

                    response = self.model.generate_content(prompt)
                    response_text = response.text.strip()
                    
                    if "```json" in response_text:
                        response_text = response_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in response_text:
                        response_text = response_text.split("```")[1].split("```")[0].strip()
                    
                    batch_classifications = json.loads(response_text)
                    classifications.update(batch_classifications)
                    
                    print(f"    ✓ Batch {batch_num}/{total_batches}: Classified {len(batch_classifications)} terms")
                    batch_success = True
                    break
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    is_rate_limit = any(keyword in error_msg for keyword in 
                                    ['rate limit', 'quota', 'too many requests', '429', 'resource exhausted'])
                    
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * initial_delay
                        
                        if is_rate_limit:
                            print(f"    ⚠️  Batch {batch_num}: Rate limit hit (attempt {attempt+1}/{max_retries})")
                            print(f"       Waiting {wait_time:.1f}s before retry...")
                        else:
                            print(f"    ⚠️  Batch {batch_num}: Error - {e} (attempt {attempt+1}/{max_retries})")
                            print(f"       Retrying in {wait_time:.1f}s...")
                        
                        time.sleep(wait_time)
                    else:
                        print(f"    ✗ Batch {batch_num}: FAILED after {max_retries} attempts")
                        print(f"       Error: {e}")
                        print(f"       Marking {len(batch)} terms as 'unknown'")
                        for term in batch:
                            classifications[term] = "unknown"
            
            if batch_success and i + batch_size < len(terms_list):
                time.sleep(initial_delay)
        
        print(f"  ✓ Gemini classification complete: {len(classifications)} terms classified")
        
        for term, classification in classifications.items():
            if classification == "translatable":
                self.translatable_terms.add(term)
            elif classification == "untranslatable":
                self.untranslatable_terms.add(term)
        
        self.gemini_classifications = classifications
        return classifications
    
    def is_translatable(self, text: str) -> bool:
        if not text or not isinstance(text, str):
            return True
        text_lower = text.lower()
        for term in self.untranslatable_terms:
            if term in text_lower:
                return False
        return True
    
    def _track_terms_in_cell(self, cell: str) -> None:
        if not cell or not isinstance(cell, str):
            return
        
        cell_lower = cell.lower()
        words = re.findall(r'\b[a-z_]+\b', cell_lower)
        
        for word in words:
            if len(word) <= 2:
                continue
            
            if word in self.translatable_terms:
                self.found_translatable_terms[word] = self.found_translatable_terms.get(word, 0) + 1
            elif word in self.untranslatable_terms:
                self.found_untranslatable_terms[word] = self.found_untranslatable_terms.get(word, 0) + 1
            else:
                found_multi = False
                for uterm in self.untranslatable_terms:
                    if ' ' in uterm and uterm in cell_lower:
                        self.found_untranslatable_terms[uterm] = self.found_untranslatable_terms.get(uterm, 0) + 1
                        found_multi = True
                        break
                
                if not found_multi:
                    if word not in ['the', 'a', 'an', 'and', 'or', 'is', 'are', 'in', 'of', 'to', 'for', 'with', 'by', 'at', 'from']:
                        self.unknown_terms[word] = self.unknown_terms.get(word, 0) + 1
    
    def has_untranslatable_label(self, text: str) -> bool:
        if not text or not isinstance(text, str):
            return False
        text_lower = text.lower()
        for term in self.untranslatable_terms:
            if term in text_lower:
                return True
        return False
    
    def count_universal_content(self, text: str) -> int:
        if not text or not isinstance(text, str):
            return 0
        
        count = 0
        count += len(re.findall(self.UNIVERSAL_ELEMENTS['numbers'], text))
        
        for op in self.UNIVERSAL_ELEMENTS['math_operators']:
            count += text.count(op)
        for symbol in self.UNIVERSAL_ELEMENTS['symbols']:
            count += text.count(symbol)
        for func in self.UNIVERSAL_ELEMENTS['math_functions']:
            count += text.lower().count(func)
        
        return count
    
    def analyze_cell_translatability(self, cell: Any) -> Tuple[float, str]:
        """Analyze a single cell for translatability."""
        
        # Handle numeric cells (binary labels, IDs, etc.) - these are universal
        if self._is_numeric_cell(cell):
            return 1.0, "numeric_universal"
        
        # Handle non-string cells
        if not isinstance(cell, str):
            return 1.0, "non_string_universal"
        
        # Handle file paths - mostly universal (paths, numbers, file extensions)
        if self._is_file_path(cell):
            # Extract only meaningful terms from the path (skip directory structure)
            filename = cell.split('/')[-1]  # Get just the filename
            cell_length = len(filename)
            universal_count = self.count_universal_content(filename)
            
            # File paths are mostly universal
            if universal_count >= cell_length * 0.3:
                return 0.95, "file_path_mostly_universal"
            return 0.85, "file_path_with_content"
        
        cell_length = len(cell)
        universal_count = self.count_universal_content(cell)
        has_untranslatable = self.has_untranslatable_label(cell)
        is_translatable = self.is_translatable(cell)
        
        self._track_terms_in_cell(cell)
        
        if has_untranslatable:
            return 0.0, "untranslatable_label"
        
        if universal_count >= cell_length * 0.5:
            return 1.0, "mostly_universal"
        
        if is_translatable and universal_count < cell_length * 0.5:
            return 0.9, "translatable_text"
        
        return 0.5, "mixed_content"
    
    def extract_translatability_analysis(self, table_data: Dict[str, Any]) -> Dict[str, Any]:
        if "data" not in table_data:
            return self._empty_analysis()
        
        data = table_data["data"]
        if len(data) <= 1:
            return self._empty_analysis()
        
        analysis = {
            'total_cells': 0,
            'cells_maintaining_meaning': 0,
            'cells_losing_meaning': 0,
            'cells_partially_translatable': 0,
            'average_translatability': 0.0,
            'untranslatable_count': 0,
            'cell_details': []
        }
        
        scores = []
        
        for row_idx, row in enumerate(data):
            for col_idx, cell in enumerate(row):
                analysis['total_cells'] += 1
                
                score, reason = self.analyze_cell_translatability(cell)
                scores.append(score)
                
                if score >= 0.7:
                    analysis['cells_maintaining_meaning'] += 1
                elif score < 0.3:
                    analysis['cells_losing_meaning'] += 1
                else:
                    analysis['cells_partially_translatable'] += 1
                
                if reason == "untranslatable_label":
                    analysis['untranslatable_count'] += 1
        
        if scores:
            analysis['average_translatability'] = round(sum(scores) / len(scores), 2)
        
        return analysis
    
    def _empty_analysis(self) -> Dict[str, Any]:
        return {
            'total_cells': 0,
            'cells_maintaining_meaning': 0,
            'cells_losing_meaning': 0,
            'cells_partially_translatable': 0,
            'average_translatability': 0.0,
            'untranslatable_count': 0,
            'cell_details': []
        }
    
    def score_table(self, table_data: Dict[str, Any]) -> int:
        analysis = self.extract_translatability_analysis(table_data)
        
        if analysis['total_cells'] == 0:
            return 1
        
        maintaining_ratio = analysis['cells_maintaining_meaning'] / analysis['total_cells']
        maintaining_score = min(10, (maintaining_ratio / 0.7) * 10)
        translatability_score = analysis['average_translatability'] * 10
        
        final_score = (maintaining_score * 0.70) + (translatability_score * 0.30)
        final_score = max(1, min(10, round(final_score)))
        
        return final_score
    
    def process_json_file(self, input_file: str) -> Dict[str, Any]:
        with open(input_file, 'r', encoding='utf-8') as f:
            table_data = json.load(f)
        
        score = self.score_table(table_data)
        
        result = {
            "file": os.path.basename(input_file),
            "score": score
        }
        
        return result
    
    def process_directory(self, directory: str, pattern: str = "*.json", max_tables: int = None) -> List[Dict[str, Any]]:
        results = []
        json_files = list(Path(directory).glob(pattern))
        
        if not json_files:
            print(f"  No JSON files found in {directory}")
            return results
        
        if max_tables is not None:
            json_files = json_files[:max_tables]
        
        # Extract and classify terms BEFORE processing tables
        if self.use_gemini:
            all_terms = self.extract_terms_from_tables(directory, pattern, max_tables)
            unknown_terms = all_terms - self.translatable_terms - self.untranslatable_terms
            if unknown_terms:
                self.classify_terms_with_gemini(unknown_terms)
        
        print(f"  Processing {len(json_files)} tables...")
        
        for json_file in json_files:
            try:
                result = self.process_json_file(str(json_file))
                results.append(result)
            except Exception as e:
                print(f"  ✗ Error processing {json_file.name}: {e}")
        
        print(f"  ✓ Completed processing {len(results)} tables")
        
        return results
    
    def calculate_score_distribution(self, results: List[Dict[str, Any]]) -> Dict[int, int]:
        """Calculate count of tables for each score (1-10)."""
        score_distribution = {i: 0 for i in range(1, 11)}
        for result in results:
            score = result['score']
            if 1 <= score <= 10:
                score_distribution[score] += 1
        return score_distribution


def main():
    """Process tables from 4 categories and save individual + combined results."""
    
    # Get Gemini API key
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    
    if not gemini_api_key:
        print("⚠️  GEMINI_API_KEY environment variable not set!")
        print("  Set it with: export GEMINI_API_KEY='your-api-key-here'")
        print("  Or add it to your .env file")
        use_gemini = input("\nContinue without Gemini API? (y/n): ").lower() == 'y'
        if not use_gemini:
            print("Exiting...")
            return
    else:
        use_gemini = True
    
    # Base directory containing all tables
    base_dir = 'data/filtered_data/filtered_tables'
    output_dir = 'src/summary'
    
    # Categories to identify (based on filename prefixes)
    categories = ['github']
    
    print(f"{'='*60}")
    print(f"MULTI-CATEGORY TABLE SCORING")
    print(f"{'='*60}\n")
    print(f"Base directory: {base_dir}")
    print(f"Categories: {', '.join(categories)}")
    print(f"Output directory: {output_dir}\n")
    
    # Check if base directory exists
    if not Path(base_dir).exists():
        print(f"❌ Error: Directory {base_dir} does not exist.")
        return
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all JSON files and categorize them
    all_files = list(Path(base_dir).glob("*.json"))
    
    if not all_files:
        print(f"❌ No JSON files found in {base_dir}")
        return
    
    print(f"Found {len(all_files)} total JSON files")
    print("Categorizing files based on filename prefixes...\n")
    
    # Categorize files based on filename
    categorized_files = {cat: [] for cat in categories}
    uncategorized = []
    
    for file in all_files:
        filename = file.name.lower()
        categorized = False
        for category in categories:
            if filename.startswith(category):
                categorized_files[category].append(file)
                categorized = True
                break
        if not categorized:
            uncategorized.append(file)
    
    # Print categorization summary
    for category in categories:
        count = len(categorized_files[category])
        print(f"  {category}: {count} files")
    
    if uncategorized:
        print(f"  uncategorized: {len(uncategorized)} files")
    
    # Create scorer instance (reuse for all categories)
    scorer = SemanticTableQualityScorer(
        gemini_api_key=gemini_api_key,
        use_gemini=use_gemini
    )
    
    # Store results for score distribution summary
    all_category_distributions = {}
    
    # Process each category
    for category in categories:
        files = categorized_files[category]
        
        print(f"\n{'='*60}")
        print(f"Processing {category.upper()} category")
        print(f"{'='*60}")
        
        if not files:
            print(f"  ⚠️  No files found for {category}. Skipping...")
            continue
        
        # Process tables in this category
        print(f"  Processing {len(files)} tables...")
        results = []

        try:
            results = scorer.process_directory(
                directory=base_dir,
                pattern=f"{category}*.json",
                max_tables=500
            )
        except Exception as e:
            print(f"  ✗ Error processing category: {e}")
        
        print(f"  ✓ Completed processing {len(results)} tables")
        
        if results:
            # Calculate statistics
            avg_score = sum(r['score'] for r in results) / len(results)
            score_distribution = scorer.calculate_score_distribution(results)
            
            # Save individual category file
            category_output = {
                "category": category,
                "scoring_method": "semantic_preservation_in_translation",
                "description": "Scores based on whether content maintains semantic meaning when translated to other languages.",
                "gemini_api_used": use_gemini,
                "total_tables": len(results),
                "average_score": round(avg_score, 2),
                "score_distribution": score_distribution,
                "tables": results
            }
            
            category_file = Path(output_dir) / f"{category}_table_scores.json"
            with open(category_file, 'w', encoding='utf-8') as f:
                json.dump(category_output, f, indent=2, ensure_ascii=False)
            
            print(f"  ✓ Saved {category} results to {category_file}")
            print(f"    Total tables: {len(results)}")
            print(f"    Average score: {avg_score:.2f}/10")
            
            # Store for combined distribution file
            all_category_distributions[category] = {
                "total_tables": len(results),
                "average_score": round(avg_score, 2),
                "score_distribution": score_distribution
            }
        else:
            print(f"  No tables processed for {category}")
    
    # Save combined score distribution file
    if all_category_distributions:
        distribution_file = Path(output_dir) / "score_distribution_summary.json"
        distribution_output = {
            "scoring_method": "semantic_preservation_in_translation",
            "description": "Score distribution summary across all categories",
            "gemini_api_used": use_gemini,
            "categories": all_category_distributions
        }
        
        with open(distribution_file, 'w', encoding='utf-8') as f:
            json.dump(distribution_output, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"\n✓ Score distribution summary saved to {distribution_file}\n")
        
        # Print summary for each category
        print("Individual category files created:")
        for category, data in all_category_distributions.items():
            print(f"  • {output_dir}/{category}_table_scores.json")
            print(f"    Tables: {data['total_tables']} | Avg Score: {data['average_score']}/10")
        
        print(f"\nCombined distribution file:")
        print(f"  • {distribution_file}")
        
        print("\n✓ All processing complete!")
    else:
        print("\n❌ No categories were processed.")


if __name__ == "__main__":
    main()
















































# import json
# import os
# from pathlib import Path
# from typing import Dict, List, Any, Tuple, Set
# import re
# import time
# from dotenv import load_dotenv

# load_dotenv()

# try:
#     import google.generativeai as genai
#     GEMINI_AVAILABLE = True
# except ImportError:
#     GEMINI_AVAILABLE = False
#     print("⚠️  Warning: google-generativeai not installed. Run: pip install google-generativeai")


# class SemanticTableQualityScorer:
#     """
#     Scores table quality based on semantic preservation across language translation.
#     Uses Gemini API to dynamically classify terms as translatable or untranslatable.
#     """
    
#     # Fallback terms if API is unavailable
#     FALLBACK_TRANSLATABLE_TERMS = {
#         'portfolio', 'strategy', 'return', 'risk', 'price', 'value',
#         'investment', 'asset', 'performance', 'ratio', 'metric', 'comparison',
#         'equal', 'contribution', 'distance', 'similarity', 'probability',
#         'distribution', 'optimization', 'minimization', 'maximization',
#         'cost', 'loss', 'gain', 'change', 'increase', 'decrease',
#         'model', 'prediction', 'forecast', 'trend', 'pattern', 'cluster',
#         'classification', 'regression', 'analysis', 'method', 'algorithm',
#         'accuracy', 'precision', 'recall', 'f1', 'score', 'rate', 'time',
#         'dataset', 'evaluation', 'benchmark', 'performance', 'result',
#         'parameter', 'coefficient', 'variable', 'feature', 'input', 'output',
#         'mean', 'standard', 'deviation', 'average', 'median', 'variance',
#         'correlation', 'relationship', 'comparison', 'difference', 'similarity'
#     }
    
#     FALLBACK_UNTRANSLATABLE_TERMS = {
#         'logistic regression', 'neural network', 'support vector', 'decision tree',
#         'activation function', 'neuron', 'layer', 'kernel',
#         'backpropagation', 'gradient descent', 'epoch', 'batch', 'overfitting',
#         'pretrained llm', 'bloom', 'gpt', 'llm', 'token', 'finetuning',
#         'pretraining', 'embedding', 'attention', 'transformer', 'lstm', 'rnn',
#         'cnn', 'gan', 'vae', 'bert', 'roberta', 'xlnet', 'electra',
#         'autoencoder', 'boltzmann', 'hopfield', 'perceptron', 'convolution'
#     }
    
#     UNIVERSAL_ELEMENTS = {
#         'numbers': r'\d+\.?\d*',
#         'math_operators': ['+', '-', '*', '/', '=', '<', '>', '≤', '≥', '^'],
#         'symbols': ['α', 'β', 'γ', 'δ', 'σ', 'λ', 'θ', 'μ', 'Δ', 'Σ', '%', '·'],
#         'math_functions': ['sin', 'cos', 'tan', 'log', 'exp', 'sqrt', 'min', 'max']
#     }
    
#     def __init__(self, gemini_api_key: str = None, use_gemini: bool = True, cache_file: str = "term_cache.json"):
#         self.use_gemini = use_gemini and GEMINI_AVAILABLE and gemini_api_key
#         self.cache_file = cache_file
#         self._load_cache()
        
#         # Initialize Gemini if available
#         if self.use_gemini:
#             genai.configure(api_key=gemini_api_key)
#             self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
#             print("✓ Gemini API initialized")
#         else:
#             self.model = None
#             if use_gemini and not GEMINI_AVAILABLE:
#                 print("⚠️  Gemini API requested but library not available. Using fallback terms.")
#             elif use_gemini and not gemini_api_key:
#                 print("⚠️  Gemini API key not provided. Using fallback terms.")
        
#         # Dynamic term classification
#         self.translatable_terms = set(self.FALLBACK_TRANSLATABLE_TERMS)
#         self.untranslatable_terms = set(self.FALLBACK_UNTRANSLATABLE_TERMS)
        
#         # Tracking
#         self.found_translatable_terms = {}
#         self.found_untranslatable_terms = {}
#         self.unknown_terms = {}
#         self.gemini_classifications = {}
#         self.terms_to_classify = set()

#     def _load_cache(self):
#         """Load previously classified terms from cache."""
#         if os.path.exists(self.cache_file):
#             with open(self.cache_file, 'r') as f:
#                 cache = json.load(f)
#                 self.translatable_terms.update(cache.get('translatable', []))
#                 self.untranslatable_terms.update(cache.get('untranslatable', []))
#                 self.gemini_classifications.update(cache.get('all_classifications', {}))
#             print(f"✓ Loaded {len(self.gemini_classifications)} cached classifications")

#     def _save_cache(self):
#         """Save classifications to cache."""
#         cache = {
#             'translatable': list(self.translatable_terms),
#             'untranslatable': list(self.untranslatable_terms),
#             'all_classifications': self.gemini_classifications
#         }
#         with open(self.cache_file, 'w') as f:
#             json.dump(cache, f, indent=2)
#         print(f"✓ Saved cache with {len(self.gemini_classifications)} classifications")
    
#     def extract_terms_from_tables(self, directory: str, pattern: str = "*.json", max_tables: int = None) -> Set[str]:
#         """Extract all unique terms from tables before processing."""
#         all_terms = set()
#         json_files = list(Path(directory).glob(pattern))
        
#         if max_tables is not None:
#             json_files = json_files[:max_tables]
        
#         print(f"  Extracting terms from {len(json_files)} tables...")
        
#         for json_file in json_files:
#             try:
#                 with open(json_file, 'r', encoding='utf-8') as f:
#                     table_data = json.load(f)
                
#                 if "data" in table_data:
#                     for row in table_data["data"]:
#                         for cell in row:
#                             if isinstance(cell, str):
#                                 words = re.findall(r'\b[a-z_]+\b', cell.lower())
#                                 all_terms.update(words)
                                
#                                 cell_lower = cell.lower()
#                                 phrases = re.findall(r'\b[a-z]+\s+[a-z]+(?:\s+[a-z]+)?\b', cell_lower)
#                                 all_terms.update(phrases)
#             except Exception as e:
#                 print(f"  Error extracting from {json_file.name}: {e}")
        
#         filtered_terms = {
#             term for term in all_terms 
#             if len(term) > 2 and term not in [
#                 'the', 'a', 'an', 'and', 'or', 'is', 'are', 'was', 'were',
#                 'in', 'of', 'to', 'for', 'with', 'by', 'at', 'from', 'as',
#                 'on', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
#                 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that'
#             ]
#         }
        
#         print(f"  ✓ Extracted {len(filtered_terms)} unique terms")
#         return filtered_terms
    
#     def classify_terms_with_gemini(self, terms: Set[str], batch_size: int = 1000, 
#                                max_retries: int =7, initial_delay: float = 0.5) -> Dict[str, str]:
#         """Use Gemini API to classify terms with robust retry logic for rate limits."""
#         if not self.use_gemini:
#             return {}
        
#         classifications = {}
#         terms_list = list(terms)
#         total_batches = (len(terms_list) + batch_size - 1) // batch_size
        
#         print(f"  Classifying {len(terms_list)} terms using Gemini API ({total_batches} batches)...")
#         print(f"  Max retries per batch: {max_retries}, Initial delay: {initial_delay}s")
        
#         for i in range(0, len(terms_list), batch_size):
#             batch = terms_list[i:i+batch_size]
#             batch_num = (i // batch_size) + 1
#             batch_success = False
            
#             # Retry loop for this batch
#             for attempt in range(max_retries):
#                 try:
#                     prompt = f"""Classify each term for cross-language semantic preservation.

# TRANSLATABLE = Universal concepts with clear, meaningful translations in other languages
# Examples: 
#   - General concepts: "weather", "climate", "disaster", "environment", "temperature", "policy", "insurance"
#   - Common tech terms: "data", "index", "table", "database", "timeseries", "attributes", "variable"
#   - Business terms: "portfolio", "strategy", "return", "risk", "accuracy", "model", "service"
#   - Legal/privacy: "privacy", "information", "user", "account", "consent", "legal"

# UNTRANSLATABLE = Specialized ML/AI jargon or technical model names that lose precise meaning
# Examples: 
#   - ML/AI terms: "LSTM", "backpropagation", "GPT-4", "BERT", "transformer", "neural network", "gradient descent"
#   - Very technical: "logistic regression", "convolutional", "recurrent neural network"

# IMPORTANT GUIDELINES:
# 1. Common technical/domain terms ARE translatable: "weather", "climate", "disaster", "emissions", "database", "index"
# 2. Business and legal terms ARE translatable: "service", "policy", "insurance", "privacy", "information"
# 3. Only mark as untranslatable if it's specialized ML/AI terminology or loses technical precision in translation
# 4. Acronyms for organizations (FEMA, NOAA) can be translatable - they're proper nouns but meaning is preserved

# Terms to classify:
# {json.dumps(batch, indent=2)}

# Respond ONLY with a JSON object mapping each term to either "translatable" or "untranslatable". Example:
# {{
#   "weather": "translatable",
#   "lstm": "untranslatable",
#   "disaster": "translatable",
#   "backpropagation": "untranslatable",
#   "insurance": "translatable"
# }}

# JSON response:"""
#                     # prompt = f"""Classify each term for cross-language semantic preservation in general text.

#                     # TRANSLATABLE = Universal concepts understood across languages
#                     # Examples: "privacy", "service", "information", "account", "user", "email", "password", "device", "analytics", "cookies"
#                     # These maintain meaning when translated.

#                     # UNTRANSLATABLE = Technical ML/AI jargon or very specific proper nouns
#                     # Examples: "LSTM", "backpropagation", "GPT-4", "BERT", "neural network"
#                     # These lose precise technical meaning in translation.

#                     # IMPORTANT: 
#                     # - Brand names like "Instagram" are proper nouns (don't translate but meaning is preserved)
#                     # - Common tech terms like "cookies", "analytics", "service" ARE translatable
#                     # - Legal/privacy terms like "privacy", "policy", "consent" ARE translatable
#                     # - Only mark as untranslatable if it's specialized ML/AI terminology

#                     # Terms to classify:
#                     # {json.dumps(batch, indent=2)}

#                     # Respond ONLY with JSON:
#                     # {{
#                     # "privacy": "translatable",
#                     # "lstm": "untranslatable",
#                     # "service": "translatable",
#                     # "instagram": "translatable"
#                     # }}

#                     # JSON response:"""
#     #                 prompt = f"""Classify each of the following terms as either "translatable" or "untranslatable" for cross-language semantic preservation.

#     # TRANSLATABLE = The term has a universal concept that can be meaningfully translated to other languages (e.g., "portfolio", "strategy", "return", "risk", "accuracy", "model")

#     # UNTRANSLATABLE = The term is domain-specific jargon or a proper noun that loses meaning when translated (e.g., "logistic regression", "neural network", "BERT", "GPT", "LSTM", "backpropagation")

#     # Terms to classify:
#     # {json.dumps(batch, indent=2)}

#     # Respond ONLY with a JSON object mapping each term to either "translatable" or "untranslatable". Example:
#     # {{
#     # "portfolio": "translatable",
#     # "lstm": "untranslatable",
#     # "accuracy": "translatable"
#     # }}

#     # JSON response:"""

#                     response = self.model.generate_content(prompt)
#                     response_text = response.text.strip()
                    
#                     # Clean JSON response
#                     if "```json" in response_text:
#                         response_text = response_text.split("```json")[1].split("```")[0].strip()
#                     elif "```" in response_text:
#                         response_text = response_text.split("```")[1].split("```")[0].strip()
                    
#                     batch_classifications = json.loads(response_text)
#                     classifications.update(batch_classifications)
                    
#                     print(f"    ✓ Batch {batch_num}/{total_batches}: Classified {len(batch_classifications)} terms")
#                     batch_success = True
#                     break  # Success! Exit retry loop
                    
#                 except Exception as e:
#                     error_msg = str(e).lower()
                    
#                     # Detect rate limit errors
#                     is_rate_limit = any(keyword in error_msg for keyword in 
#                                     ['rate limit', 'quota', 'too many requests', '429', 'resource exhausted'])
                    
#                     if attempt < max_retries - 1:  # Not the last attempt
#                         # Calculate exponential backoff delay
#                         wait_time = (2 ** attempt) * initial_delay
                        
#                         if is_rate_limit:
#                             print(f"    ⚠️  Batch {batch_num}: Rate limit hit (attempt {attempt+1}/{max_retries})")
#                             print(f"       Waiting {wait_time:.1f}s before retry...")
#                         else:
#                             print(f"    ⚠️  Batch {batch_num}: Error - {e} (attempt {attempt+1}/{max_retries})")
#                             print(f"       Retrying in {wait_time:.1f}s...")
                        
#                         time.sleep(wait_time)
                        
#                     else:  # Last attempt failed
#                         print(f"    ❌ Batch {batch_num}: FAILED after {max_retries} attempts")
#                         print(f"       Error: {e}")
#                         print(f"       Marking {len(batch)} terms as 'unknown'")
#                         # Only mark as unknown if ALL retries failed
#                         for term in batch:
#                             classifications[term] = "unknown"
            
#             # Add delay between batches (only after successful batch)
#             if batch_success and i + batch_size < len(terms_list):  # Not the last batch
#                 time.sleep(initial_delay)
        
#         print(f"  ✓ Gemini classification complete: {len(classifications)} terms classified")
        
#         # Update term sets
#         for term, classification in classifications.items():
#             if classification == "translatable":
#                 self.translatable_terms.add(term)
#             elif classification == "untranslatable":
#                 self.untranslatable_terms.add(term)
        
#         self.gemini_classifications = classifications
#         return classifications
    
# #     def classify_terms_with_gemini(self, terms: Set[str], batch_size: int = 100) -> Dict[str, str]:
# #         """Use Gemini API to classify terms as translatable or untranslatable."""
# #         if not self.use_gemini:
# #             return {}
        
# #         classifications = {}
# #         terms_list = list(terms)
# #         total_batches = (len(terms_list) + batch_size - 1) // batch_size
        
# #         print(f"  Classifying {len(terms_list)} terms using Gemini API ({total_batches} batches)...")
        
# #         for i in range(0, len(terms_list), batch_size):
# #             batch = terms_list[i:i+batch_size]
# #             batch_num = (i // batch_size) + 1
            
# #             try:
# #                 prompt = f"""Classify each of the following terms as either "translatable" or "untranslatable" for cross-language semantic preservation.

# # TRANSLATABLE = The term has a universal concept that can be meaningfully translated to other languages (e.g., "portfolio", "strategy", "return", "risk", "accuracy", "model")

# # UNTRANSLATABLE = The term is domain-specific jargon or a proper noun that loses meaning when translated (e.g., "logistic regression", "neural network", "BERT", "GPT", "LSTM", "backpropagation")

# # Terms to classify:
# # {json.dumps(batch, indent=2)}

# # Respond ONLY with a JSON object mapping each term to either "translatable" or "untranslatable". Example:
# # {{
# #   "portfolio": "translatable",
# #   "lstm": "untranslatable",
# #   "accuracy": "translatable"
# # }}

# # JSON response:"""

# #                 response = self.model.generate_content(prompt)
# #                 response_text = response.text.strip()
                
# #                 if "```json" in response_text:
# #                     response_text = response_text.split("```json")[1].split("```")[0].strip()
# #                 elif "```" in response_text:
# #                     response_text = response_text.split("```")[1].split("```")[0].strip()
                
# #                 batch_classifications = json.loads(response_text)
# #                 classifications.update(batch_classifications)
                
# #                 print(f"    Batch {batch_num}/{total_batches}: Classified {len(batch_classifications)} terms")
                
# #                 time.sleep(0.5) # To respect rate limits   
                
# #             except Exception as e:
# #                 print(f"    ⚠️  Error in batch {batch_num}: {e}")
# #                 for term in batch:
# #                     classifications[term] = "unknown"
        
# #         print(f"  ✓ Gemini classification complete: {len(classifications)} terms classified")
        
# #         for term, classification in classifications.items():
# #             if classification == "translatable":
# #                 self.translatable_terms.add(term)
# #             elif classification == "untranslatable":
# #                 self.untranslatable_terms.add(term)
        
# #         self.gemini_classifications = classifications
# #         return classifications
    
#     def is_translatable(self, text: str) -> bool:
#         if not text or not isinstance(text, str):
#             return True
#         text_lower = text.lower()
#         for term in self.untranslatable_terms:
#             if term in text_lower:
#                 return False
#         return True
    
#     def _track_terms_in_cell(self, cell: str) -> None:
#         if not cell or not isinstance(cell, str):
#             return
        
#         cell_lower = cell.lower()
#         words = re.findall(r'\b[a-z_]+\b', cell_lower)
        
#         for word in words:
#             if len(word) <= 2:
#                 continue
            
#             if word in self.translatable_terms:
#                 self.found_translatable_terms[word] = self.found_translatable_terms.get(word, 0) + 1
#             elif word in self.untranslatable_terms:
#                 self.found_untranslatable_terms[word] = self.found_untranslatable_terms.get(word, 0) + 1
#             else:
#                 found_multi = False
#                 for uterm in self.untranslatable_terms:
#                     if ' ' in uterm and uterm in cell_lower:
#                         self.found_untranslatable_terms[uterm] = self.found_untranslatable_terms.get(uterm, 0) + 1
#                         found_multi = True
#                         break
                
#                 if not found_multi:
#                     if word not in ['the', 'a', 'an', 'and', 'or', 'is', 'are', 'in', 'of', 'to', 'for', 'with', 'by', 'at', 'from']:
#                         self.unknown_terms[word] = self.unknown_terms.get(word, 0) + 1
    
#     def has_untranslatable_label(self, text: str) -> bool:
#         if not text or not isinstance(text, str):
#             return False
#         text_lower = text.lower()
#         for term in self.untranslatable_terms:
#             if term in text_lower:
#                 return True
#         return False
    
#     def count_universal_content(self, text: str) -> int:
#         if not text or not isinstance(text, str):
#             return 0
        
#         count = 0
#         count += len(re.findall(self.UNIVERSAL_ELEMENTS['numbers'], text))
        
#         for op in self.UNIVERSAL_ELEMENTS['math_operators']:
#             count += text.count(op)
#         for symbol in self.UNIVERSAL_ELEMENTS['symbols']:
#             count += text.count(symbol)
#         for func in self.UNIVERSAL_ELEMENTS['math_functions']:
#             count += text.lower().count(func)
        
#         return count
    
#     def analyze_cell_translatability(self, cell: str) -> Tuple[float, str]:
#         if not cell or not isinstance(cell, str):
#             return 1.0, "empty_cell"

#         # ADD THIS: Detect URLs
#         if cell.startswith(('http://', 'https://', 'www.', 'ftp://')):
#             return 1.0, "url_universal"
        
#         cell_length = len(cell)
#         universal_count = self.count_universal_content(cell)
#         has_untranslatable = self.has_untranslatable_label(cell)
#         is_translatable = self.is_translatable(cell)
        
#         self._track_terms_in_cell(cell)
        
#         if has_untranslatable:
#             return 0.0, "untranslatable_label"
        
#         if universal_count >= cell_length * 0.5:
#             return 1.0, "mostly_universal"
        
#         if is_translatable and universal_count < cell_length * 0.5:
#             return 0.9, "translatable_text"
        
#         return 0.5, "mixed_content"
    
#     def extract_translatability_analysis(self, table_data: Dict[str, Any]) -> Dict[str, Any]:
#         if "data" not in table_data:
#             return self._empty_analysis()
        
#         data = table_data["data"]
#         if len(data) <= 1:
#             return self._empty_analysis()
        
#         analysis = {
#             'total_cells': 0,
#             'cells_maintaining_meaning': 0,
#             'cells_losing_meaning': 0,
#             'cells_partially_translatable': 0,
#             'average_translatability': 0.0,
#             'untranslatable_count': 0,
#             'cell_details': []
#         }
        
#         scores = []
        
#         for row_idx, row in enumerate(data):
#             for col_idx, cell in enumerate(row):
#                 analysis['total_cells'] += 1
                
#                 if isinstance(cell, str):
#                     score, reason = self.analyze_cell_translatability(cell)
#                     scores.append(score)
                    
#                     if score >= 0.7:
#                         analysis['cells_maintaining_meaning'] += 1
#                     elif score < 0.3:
#                         analysis['cells_losing_meaning'] += 1
#                     else:
#                         analysis['cells_partially_translatable'] += 1
                    
#                     if reason == "untranslatable_label":
#                         analysis['untranslatable_count'] += 1
        
#         if scores:
#             analysis['average_translatability'] = round(sum(scores) / len(scores), 2)
        
#         return analysis
    
#     def _empty_analysis(self) -> Dict[str, Any]:
#         return {
#             'total_cells': 0,
#             'cells_maintaining_meaning': 0,
#             'cells_losing_meaning': 0,
#             'cells_partially_translatable': 0,
#             'average_translatability': 0.0,
#             'untranslatable_count': 0,
#             'cell_details': []
#         }
    
#     def score_table(self, table_data: Dict[str, Any]) -> int:
#         analysis = self.extract_translatability_analysis(table_data)

#         print(f"DEBUG - Total cells: {analysis['total_cells']}")
#         print(f"DEBUG - Maintaining: {analysis['cells_maintaining_meaning']}")
#         print(f"DEBUG - Losing: {analysis['cells_losing_meaning']}")
#         print(f"DEBUG - Avg translatability: {analysis['average_translatability']}")
        
#         if analysis['total_cells'] == 0:
#             return 1
        
#         maintaining_ratio = analysis['cells_maintaining_meaning'] / analysis['total_cells']
#         maintaining_score = min(10, (maintaining_ratio / 0.7) * 10)
#         translatability_score = analysis['average_translatability'] * 10
        
#         final_score = (maintaining_score * 0.70) + (translatability_score * 0.30)
#         final_score = max(1, min(10, round(final_score)))
        
#         return final_score
    
#     def process_json_file(self, input_file: str) -> Dict[str, Any]:
#         with open(input_file, 'r', encoding='utf-8') as f:
#             table_data = json.load(f)
        
#         score = self.score_table(table_data)
        
#         result = {
#             "file": os.path.basename(input_file),
#             "score": score
#         }
        
#         return result
    
#     def process_directory(self, directory: str, pattern: str = "*.json", max_tables: int = None) -> List[Dict[str, Any]]:
#         results = []
#         json_files = list(Path(directory).glob(pattern))
        
#         if not json_files:
#             print(f"  No JSON files found in {directory}")
#             return results
        
#         if max_tables is not None:
#             json_files = json_files[:max_tables]
        
#         # Extract and classify terms BEFORE processing tables
#         if self.use_gemini:
#             all_terms = self.extract_terms_from_tables(directory, pattern, max_tables)
#             unknown_terms = all_terms - self.translatable_terms - self.untranslatable_terms
#             if unknown_terms:
#                 self.classify_terms_with_gemini(unknown_terms)
        
#         print(f"  Processing {len(json_files)} tables...")
        
#         for json_file in json_files:
#             try:
#                 result = self.process_json_file(str(json_file))
#                 results.append(result)
#             except Exception as e:
#                 print(f"  ✗ Error processing {json_file.name}: {e}")
        
#         print(f"  ✓ Completed processing {len(results)} tables")
        
#         return results
    
#     def calculate_score_distribution(self, results: List[Dict[str, Any]]) -> Dict[int, int]:
#         """Calculate count of tables for each score (1-10)."""
#         score_distribution = {i: 0 for i in range(1, 11)}
#         for result in results:
#             score = result['score']
#             if 1 <= score <= 10:
#                 score_distribution[score] += 1
#         return score_distribution


# def main():
#     """Process tables from 4 categories and save individual + combined results."""
    
#     # Get Gemini API key
#     gemini_api_key = os.getenv('GEMINI_API_KEY')
    
#     if not gemini_api_key:
#         print("⚠️  GEMINI_API_KEY environment variable not set!")
#         print("  Set it with: export GEMINI_API_KEY='your-api-key-here'")
#         print("  Or add it to your .env file")
#         use_gemini = input("\nContinue without Gemini API? (y/n): ").lower() == 'y'
#         if not use_gemini:
#             print("Exiting...")
#             return
#     else:
#         use_gemini = True
    
#     # Base directory containing all tables
#     # base_dir = 'sample_dat'
#     base_dir = 'data/filtered_data/filtered_tables'
#     output_dir = 'src/summary'
    
#     # Categories to identify (based on filename prefixes)
#     # categories = ['arxiv', 'finqa', 'github', 'wikisql']
#     categories = ['github']
    
#     print(f"{'='*60}")
#     print(f"MULTI-CATEGORY TABLE SCORING")
#     print(f"{'='*60}\n")
#     print(f"Base directory: {base_dir}")
#     print(f"Categories: {', '.join(categories)}")
#     print(f"Output directory: {output_dir}\n")
    
#     # Check if base directory exists
#     if not Path(base_dir).exists():
#         print(f"❌ Error: Directory {base_dir} does not exist.")
#         return
    
#     # Create output directory
#     Path(output_dir).mkdir(parents=True, exist_ok=True)
    
#     # Get all JSON files and categorize them
#     all_files = list(Path(base_dir).glob("*.json"))
    
#     if not all_files:
#         print(f"❌ No JSON files found in {base_dir}")
#         return
    
#     print(f"Found {len(all_files)} total JSON files")
#     print("Categorizing files based on filename prefixes...\n")
    
#     # Categorize files based on filename
#     categorized_files = {cat: [] for cat in categories}
#     uncategorized = []
    
#     for file in all_files:
#         filename = file.name.lower()
#         categorized = False
#         for category in categories:
#             if filename.startswith(category):
#                 categorized_files[category].append(file)
#                 categorized = True
#                 break
#         if not categorized:
#             uncategorized.append(file)
    
#     # Print categorization summary
#     for category in categories:
#         count = len(categorized_files[category])
#         print(f"  {category}: {count} files")
    
#     if uncategorized:
#         print(f"  uncategorized: {len(uncategorized)} files")
    
#     # Create scorer instance (reuse for all categories)
#     scorer = SemanticTableQualityScorer(
#         gemini_api_key=gemini_api_key,
#         use_gemini=use_gemini
#     )
    
#     # Store results for score distribution summary
#     all_category_distributions = {}
    
#     # Process each category
#     for category in categories:
#         files = categorized_files[category]
        
#         print(f"\n{'='*60}")
#         print(f"Processing {category.upper()} category")
#         print(f"{'='*60}")
        
#         if not files:
#             print(f"  ⚠️  No files found for {category}. Skipping...")
#             continue
        
#         # Process tables in this category
#         print(f"  Processing {len(files)} tables...")
#         results = []

#         try:
#             # Use the existing process_directory method with max_tables support
#             results = scorer.process_directory(
#                 directory=base_dir,
#                 pattern=f"{category}*.json",
#                 max_tables=500  # <-- Could add limit here
#             )
#         except Exception as e:
#             print(f"  ✗ Error processing {category_file.name}: {e}")
        
#         print(f"  ✓ Completed processing {len(results)} tables")
        
#         if results:
#             # Calculate statistics
#             avg_score = sum(r['score'] for r in results) / len(results)
#             score_distribution = scorer.calculate_score_distribution(results)
            
#             # Save individual category file
#             category_output = {
#                 "category": category,
#                 "scoring_method": "semantic_preservation_in_translation",
#                 "description": "Scores based on whether content maintains semantic meaning when translated to other languages.",
#                 "gemini_api_used": use_gemini,
#                 "total_tables": len(results),
#                 "average_score": round(avg_score, 2),
#                 "score_distribution": score_distribution,
#                 "tables": results
#             }
            
#             category_file = Path(output_dir) / f"{category}_table_scores.json"
#             with open(category_file, 'w', encoding='utf-8') as f:
#                 json.dump(category_output, f, indent=2, ensure_ascii=False)
            
#             print(f"  ✓ Saved {category} results to {category_file}")
#             print(f"    Total tables: {len(results)}")
#             print(f"    Average score: {avg_score:.2f}/10")
            
#             # Store for combined distribution file
#             all_category_distributions[category] = {
#                 "total_tables": len(results),
#                 "average_score": round(avg_score, 2),
#                 "score_distribution": score_distribution
#             }
#         else:
#             print(f"  No tables processed for {category}")
    
#     # Save combined score distribution file
#     if all_category_distributions:
#         distribution_file = Path(output_dir) / "score_distribution_summary.json"
#         distribution_output = {
#             "scoring_method": "semantic_preservation_in_translation",
#             "description": "Score distribution summary across all categories",
#             "gemini_api_used": use_gemini,
#             "categories": all_category_distributions
#         }
        
#         with open(distribution_file, 'w', encoding='utf-8') as f:
#             json.dump(distribution_output, f, indent=2, ensure_ascii=False)
        
#         print(f"\n{'='*60}")
#         print(f"FINAL SUMMARY")
#         print(f"{'='*60}")
#         print(f"\n✓ Score distribution summary saved to {distribution_file}\n")
        
#         # Print summary for each category
#         print("Individual category files created:")
#         for category, data in all_category_distributions.items():
#             print(f"  • {output_dir}/{category}_table_scores.json")
#             print(f"    Tables: {data['total_tables']} | Avg Score: {data['average_score']}/10")
        
#         print(f"\nCombined distribution file:")
#         print(f"  • {distribution_file}")
        
#         print("\n✓ All processing complete!")
#     else:
#         print("\n❌ No categories were processed.")


# if __name__ == "__main__":
#     main()
