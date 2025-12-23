import asyncio
import logging
import os
from datetime import datetime
from typing import List, Optional

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import context_precision, context_recall

from utils import fetch_all_answers_in_batches

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
SAMPLE_PATH = os.getenv("SAMPLE_PATH", "data/retrieval_evaluation_dataset.csv")
RESULT_DIR = os.getenv("RESULT_DIR", "data")
EVAL_MODEL = os.getenv("EVALUATION_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Initialize RAGAS models
llm = ChatOpenAI(model=EVAL_MODEL)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)


def run_ragas_context_eval() -> None:
    """
    Run RAGAS evaluation for context precision and recall using golden reference contexts.
    """
    os.makedirs(RESULT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    result_path = os.path.join(RESULT_DIR, f"ragas_context_results_{timestamp}.csv")

    try:
        samples_df = pd.read_csv(SAMPLE_PATH)
        logging.info(f"Loaded {len(samples_df)} samples from {SAMPLE_PATH}")
    except FileNotFoundError:
        logging.error(f"Sample file not found at {SAMPLE_PATH}")
        return

    # Fetch answers and contexts asynchronously
    eval_data = asyncio.run(fetch_all_answers_in_batches(samples_df, batch_size=10))
    eval_df = pd.DataFrame(eval_data)

    if eval_df.empty:
        logging.warning("No data fetched for evaluation.")
        return

    # Create dataset for RAGAS
    dataset = Dataset.from_pandas(eval_df[["question", "contexts", "reference"]])

    metrics = [context_precision, context_recall]

    logging.info("Running RAGAS context evaluation...")
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )

    logging.info("=== RAGAS Context Evaluation Results ===")
    logging.info(results)

    # Combine results with original questions for analysis
    results_df = results.to_pandas()
    out_df = pd.concat(
        [eval_df[["question"]].reset_index(drop=True), results_df.reset_index(drop=True)], 
        axis=1
    )

    # Calculate and append average metrics
    metric_cols = [m.name for m in metrics]
    avg_scores = out_df[metric_cols].mean(numeric_only=True)
    
    avg_row = {"question": "AVERAGE"}
    for col in metric_cols:
         avg_row[col] = avg_scores.get(col)

    out_df = pd.concat([out_df, pd.DataFrame([avg_row])], ignore_index=True)

    # Save results
    out_df.to_csv(result_path, index=False)
    logging.info(f"Detailed results saved to {result_path}")


if __name__ == "__main__":
    run_ragas_context_eval()
