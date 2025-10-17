from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

TABLES_DIR = PROCESSED_DATA_DIR / "tables"


QA_PAIRS_DIR = PROCESSED_DATA_DIR / "qa_pairs"
QA_PAIRS_DIR.mkdir(parents=True, exist_ok=True)

NUM_QUESTIONS_PER_TABLE = 10  

REASONING_CATEGORIES = {
    "1. Comparative Reasoning": "Questions that require comparing multiple rows or values to find a relative answer (e.g., 'Which product had the highest profit margin?').",
    "2. Numerical Aggregation": "Questions needing calculations across several cells, like SUM, AVERAGE, COUNT, or MEDIAN (e.g., 'What was the total revenue for all quarters in 2023?').",
    "3. Multi-Hop Reasoning": "Complex questions where the answer to a hidden sub-question is needed to answer the main question (e.g., 'What is the name of the manager in the department with the lowest average employee tenure?').",
    "4. Temporal Reasoning": "Questions about trends, sequences, or durations over time-based data (e.g., 'Did the company's stock price show consistent growth between 2018 and 2022?').",
    "5. Conditional Reasoning": "Questions with logical conditions (IF-THEN, AND/OR) that filter the data (e.g., 'List all employees in the 'Sales' department who exceeded their Q3 target.').",
    "6. Proportional/Ratio Analysis": "Questions that require calculating percentages, ratios, or fractions of totals (e.g., 'What percentage of the total website traffic came from organic search?').",
    "7. Hypothetical Reasoning": "Scenarios that ask for a result based on a change to the table's data (e.g., 'If the cost for 'Material A' increased by 15%, what would the new total project cost be?').",
    "8. Correlation Inference": "Questions that ask for potential relationships or patterns between two columns (e.g., 'Is there a correlation between marketing spend and the number of new user sign-ups?').",
    "9. Structural/Metadata Reasoning": "Questions about the table's structure itself, not just its content (e.g., 'Which column contains the most missing values?').",
    "10. Outlier Detection": "Questions that require identifying data points that are significantly different from others (e.g., 'Which month had an unusually high number of server errors compared to the average?')."
}


GEMINI_API_KEYS = [
   "API"
]
GEMINI_MODEL_NAME = "gemini-2.5-pro"
