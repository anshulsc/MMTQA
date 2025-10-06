from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List, Dict, Any

def flatten_table_for_bleu(table_json: Dict[str, Any]) -> List[str]:
    content = []
    content.extend(table_json.get('columns', []))
    for row in table_json.get('data', []):
        content.extend([str(cell) for cell in row])
    return content

def calculate_bleu(reference_table: Dict, candidate_table: Dict) -> float:

    reference_tokens = flatten_table_for_bleu(reference_table)
    candidate_tokens = flatten_table_for_bleu(candidate_table)

    if not reference_tokens or not candidate_tokens:
        return 0.0

    reference_corpus = [reference_tokens]
    
    smoother = SmoothingFunction().method1
    
    score = sentence_bleu(reference_corpus, candidate_tokens, smoothing_function=smoother)
    
    return score