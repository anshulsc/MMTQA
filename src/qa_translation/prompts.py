from pydantic import BaseModel, Field
from typing import List
import json


class TranslatedQA(BaseModel):
    translated_question: str = Field(..., description="The translated version of the question.")
    translated_answer: List[List[str]] = Field(..., description="The translated version of the answer, preserving non-textual content where appropriate.")


QA_TRANSLATION_PROMPT = """
You are an expert linguist and professional translator with deep expertise in structured data. Your task is to accurately translate a question-answer pair from English to **{target_language}**.

**CONTEXT:**
The question-answer pair is based on the following data table. Use this table to understand the context of entities, numbers, and technical terms.

**Context Table:**

{context_table_json}


---
**ENGLISH QUESTION-ANSWER PAIR TO TRANSLATE:**

{english_qa_json}


---
**CRITICAL INSTRUCTIONS:**
1.  **Translate the `question`:** Convert the question text into fluent and natural-sounding **{target_language}**.
2.  **Translate the `answer` CONDITIONALLY:**
    -   If the `question_type` is **'open_ended_reasoning'**, you MUST translate the full text of the answer.
    -   If the `question_type` is **'value'**, you MUST PRESERVE the original answer exactly. DO NOT translate numbers (e.g., "370"), names (e.g., "Beta"), percentages, or codes.
3.  **Strict JSON Output:** Your entire response must be a single, valid JSON object with ONLY two keys: `translated_question` and `translated_answer`. Do not add any other text, explanations, or markdown code blocks.

---
**EXAMPLE:**

If `target_language` is **Spanish** and the input QA pair is:

{{
  "question": "Which product experienced a decline in units sold from 2022 to 2023?",
  "answer": [["Beta"]],
  "question_type": "value"
}}

Your perfect JSON output would be:
{{
  "question": "¿Qué producto experimentó una disminución en las unidades vendidas de 2022 a 2023?",
  "answer": [["Beta"]]
}}

Notice how "Beta" was NOT translated because the `question_type` was 'value'.

---
**YOUR TASK:**
Translate the provided English QA pair to **{target_language}** following all instructions.

**Your JSON Output:**
"""