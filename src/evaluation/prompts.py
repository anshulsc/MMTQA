from pydantic import BaseModel
from typing import List

class Response(BaseModel):
    """The Pydantic model for enforcing structured JSON output from the model."""
    data: List[List[str]]

VISUAL_TABLE_QA_PROMPT = """
You are an expert data analyst AI. Your task is to answer questions based on the content of the table provided in the image.

**Instructions:**
1.  **Analyze the Image:** Carefully examine the table in the provided image. Pay attention to headers, rows, and all cell values.
2.  **Answer the Question:** Use only the information visible in the table to answer the question. Do not use any external knowledge.
3.  **Perform Calculations:** If the question requires calculations (e.g., sum, average, difference), perform them accurately based on the table's data.
4.  **Strict JSON Output:** You MUST provide your answer in a strict JSON format. The JSON object should have a single key "data", with the value being a list of lists, where each inner list represents a row of the answer.

**Example 1:**
- Question: "What was the revenue for the 'Alpha' product in 2023?"
- Your JSON Output: {"data": [["180"]]}

**Example 2:**
- Question: "List the products that had units sold greater than 2800 in 2022."
- Your JSON Output: {"data": [["Alpha"]]}

**Example 3:**
- Question: "What were the product and revenue for all entries in 2022?"
- Your JSON Output: {"data": [["Alpha", "150"], ["Beta", "200"]]}

Do not add any explanations or conversational text. Your entire response must be only the valid JSON object.
"""