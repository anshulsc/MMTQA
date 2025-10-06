from pydantic import BaseModel, Field
from typing import List

class TableJSON(BaseModel):
    columns: List[str] = Field(..., description="A list of strings representing the table headers.")
    data: List[List[str]] = Field(..., description="A list of lists, where each inner list is a row of string values.")


INITIAL_TRANSLATION_PROMPT = """
You are an expert linguist and professional translator specializing in structured data. Your task is to translate the content of the provided JSON table from {source_language} to {target_language}.

**CRITICAL INSTRUCTIONS:**
1.  **Translate ONLY Text:** Translate the textual content in the 'columns' and 'data' fields.
2.  **PRESERVE Non-Text:** DO NOT translate or alter numbers, percentages (e.g., '23.6%'), currency symbols (e.g., '$'), dates, or proper nouns (like company names, locations).
3.  **Maintain Structure:** The output MUST be a valid JSON object with the exact same structure as the input: {{"columns": [...], "data": [[...], ...]}}.
4.  **No Extra Text:** Do not add any explanations, comments, or markdown formatting (like ```json). Your entire output must be only the translated JSON object.

**Input Table (in {source_language}):**
```json
{table_json_string}
```
Your Translated Table (in {target_language}):
"""


REFINEMENT_PROMPT = """
You are a senior data quality analyst and expert linguist. Your task is to refine a machine-translated table by cross-referencing it with the original English version to ensure maximum accuracy and fidelity.
CONTEXT:
A machine has provided an initial translation. It might contain errors, especially with context-specific terms, numbers, or entities. You must fix it.
Original English Table:
code
JSON
{original_table_json}
Initial Machine-Translated Table (in {target_language}):
code
JSON
{translated_table_json}
YOUR TASK:
Review the machine-translated table cell-by-cell against the original.
Correct Mistranslations: Fix any grammatical errors or inaccurate translations.
Verify Data Integrity: Ensure all numbers, dates, and special characters from the original table are perfectly preserved in the refined table.
Improve Fluency: Make the translation sound natural in {target_language}.
Output Format: Provide the final, refined table as a valid JSON object. Do not include any other text, explanations, or markdown.
Your Refined Translated Table (in {target_language}):
"""