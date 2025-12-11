import pandas as pd
import os
import logging
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from utils import fetch_all_answers_in_batches

from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    GEval
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

API_URL = os.getenv("API_URL", "http://localhost:8000/chat")
SAMPLE_PATH = os.getenv("SAMPLE_PATH", "data/golden_samples.csv")
RESULT_DIR = os.getenv("RESULT_DIR", "data")


def run_deepeval_evaluation():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    result_path = os.path.join(RESULT_DIR, f"deepeval_results_{timestamp}.csv")

    samples_df = pd.read_csv(SAMPLE_PATH)
    eval_data = asyncio.run(fetch_all_answers_in_batches(samples_df, batch_size=10))

    test_cases = []
    for row in eval_data:
        retrieval_context = row['contexts'] if isinstance(row['contexts'], list) else [str(row['contexts'])]

        test_case = LLMTestCase(
            input=row['question'],
            actual_output=row['answer'],
            expected_output=row['ground_truth'],
            retrieval_context=retrieval_context
        )
        test_cases.append(test_case)

    eval_model = "gpt-4o-mini"

    metrics = [
        FaithfulnessMetric(threshold=0.5, model=eval_model),
        AnswerRelevancyMetric(threshold=0.5, model=eval_model),
        ContextualPrecisionMetric(threshold=0.5, model=eval_model),
        ContextualRecallMetric(threshold=0.5, model=eval_model),
        GEval(
            name="Correctness",
            criteria="Determine whether the actual output answers the input request correctly, using the expected output as a reference for facts. If the actual output correctly answers the input, score it 1.0. Additional helpful information is allowed and should NOT be penalized.",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            model=eval_model
        )
    ]

    logging.info("Running DeepEval evaluation...")
    evaluate(test_cases, metrics)


if __name__ == "__main__":
    run_deepeval_evaluation()
