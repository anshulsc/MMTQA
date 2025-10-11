from pydantic import BaseModel, Field
from typing import List, Literal
from src.configs.qa_config import REASONING_CATEGORIES

ReasoningCategory = Literal[
    "Comparative Reasoning", "Numerical Aggregation", "Multi-Hop Reasoning",
    "Temporal Reasoning", "Conditional Reasoning", "Proportional/Ratio Analysis",
    "Hypothetical Reasoning", "Correlation Inference", "Structural/Metadata Reasoning",
    "Outlier Detection"
]

QuestionType = Literal["value", "open_ended_reasoning"]

class SingleQA(BaseModel):
    question: str = Field(..., description="A challenging, clear, and unambiguous question based on the table.")
    answer: List[List[str]] = Field(
        ..., 
        description="The precise answer derived from the table. For value-based questions: use exact values (e.g., [['150']] or [['Alpha'], ['Beta']]). For open-ended reasoning: use comprehensive explanation (e.g., [['The data shows...']])."
    )
    evidence_cells: List[str] = Field(..., description="A list of cell coordinates (e.g., ['B3', 'C4']) that are essential to derive the answer.")
    reasoning_category: ReasoningCategory = Field(..., description="The category of reasoning required to answer the question.")
    question_type: QuestionType = Field(..., description="Type of question: 'value' for specific data values or calculations, 'open_ended_reasoning' for analytical insights and explanations.")

class GeneratedQACollection(BaseModel):
    qa_pairs: List[SingleQA] = Field(..., description="A list of generated question-answer pairs.")

def get_category_descriptions() -> str:
    return "\n".join([f"- **{name}**: {desc}" for name, desc in REASONING_CATEGORIES.items()])


import json 


def get_output_schema() -> str:
    return json.dumps(GeneratedQACollection.model_json_schema(), indent=2)

QA_GENERATION_PROMPT = f"""
You are a world-class data analyst and expert curriculum designer. Your task is to generate a set of **{{num_questions}}** diverse, challenging, and high-quality question-answer pairs based on the provided data table in JSON format. The questions must require deep reasoning and not be simple lookups.

**CRITICAL INSTRUCTIONS:**
1.  **Generate Diverse Questions:** Create questions that cover a wide range of the reasoning categories defined below. Do not repeat question patterns.
2.  **Ensure Answerability:** Every question must be answerable *exclusively* from the provided table. Do not require external knowledge.
3.  **Provide Precise Answers:** 
    - For **value-based questions**: The 'answer' must be the exact value(s) from the table or calculated from it. Format it as a list of lists (e.g., [["150"]] or [["Alpha"], ["Beta"]]).
    - For **open-ended reasoning questions**: The 'answer' should be a comprehensive explanation or analysis based on the data, formatted as a list containing a single list with one string element (e.g., [["The data shows a declining trend because..."]]).
4.  **Specify Question Type:** Each question must have a 'question_type' field that is either "value" or "open_ended_reasoning".
5.  **Identify Evidence:** The 'evidence_cells' must accurately list all cells needed to formulate the answer. Use standard spreadsheet notation (e.g., A1 for the top-left-most cell in the data body, where Column A is the first column and Row 1 is the first data row. Headers are considered Row 0, so A1 refers to the first data cell, not a header).
6.  **Strict JSON Output:** Your response MUST be a single, valid JSON object that conforms to the provided schema. Do not include any explanatory text, markdown, or comments outside of the JSON structure, and **do not wrap the JSON in markdown code blocks (e.g., ```json)**.

**REASONING CATEGORIES TO USE:**
{get_category_descriptions()}


**REQUIRED JSON OUTPUT SCHEMA:**
Your entire response must be a single JSON object matching this schema.
{{answer_format}}


EXAMPLE OF A PERFECT OUTPUT:
Input Table (as JSON):
{{example_table}}

Question Answer:
{{qa_example}}

NOW, GENERATE QA PAIR FOR THE FOLLOWING TABLE:
Input Table (as JSON):
{{table_as_json_string}}

Your JSON Output:
"""

ANSWER_FORMAT = {
    "qa_pairs": [
        {
            "question": "string",
            "answer": "List[List[string]]",
            "evidence_cells": "List[string]",
            "reasoning_category": "string (must be one of the 10 categories)",
            "question_type": "string (either 'value' or 'open_ended_reasoning')"
        },
        ...
    ]
}

EXAMPLE_TABLE = {
  "columns": ["Year", "Product", "Revenue ($M)", "Units Sold"],
  "data": [
    ["2022", "Alpha", "150", "3000"],
    ["2022", "Beta", "200", "2500"],
    ["2023", "Alpha", "180", "3500"],
    ["2023", "Beta", "190", "2400"]
  ]
}

QA_EXAMPLE = {
  "qa_pairs": [
    {
      "question": "What was the total revenue generated across all products in 2023?",
      "answer": [["370"]],
      "evidence_cells": ["C3", "C4"],
      "reasoning_category": "Numerical Aggregation",
      "question_type": "value"
    },
    {
      "question": "Which product experienced a decline in units sold from 2022 to 2023?",
      "answer": [["Beta"]],
      "evidence_cells": ["B2", "D2", "B4", "D4"],
      "reasoning_category": "Comparative Reasoning",
      "question_type": "value"
    },
    {
      "question": "Based on the revenue and units sold trends, what strategic insights can you derive about the performance of both products?",
      "answer": [["Alpha shows strong growth with both revenue (20% increase) and units sold (16.7% increase) rising from 2022 to 2023, indicating healthy market demand and possibly effective pricing strategy. Beta, however, shows concerning signs with revenue declining by 5% and units sold dropping by 4%, suggesting potential market saturation, increased competition, or product quality issues. The company should investigate Beta's declining performance and consider reallocating resources to capitalize on Alpha's momentum."]],
      "evidence_cells": ["A1", "B1", "C1", "D1", "A2", "B2", "C2", "D2", "A3", "B3", "C3", "D3", "A4", "B4", "C4", "D4"],
      "reasoning_category": "Multi-hop Reasoning",
      "question_type": "open_ended_reasoning"
    }
  ]
}