import pandas as pd
import os
import ast
import logging
import asyncio
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
from deepeval.evaluate import AsyncConfig
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

API_URL = os.getenv("API_URL", "http://localhost:8000/chat")
SAMPLE_PATH = os.getenv("SAMPLE_PATH", "data/golden_samples.csv")
RESULT_DIR = os.getenv("RESULT_DIR", "data")


def run_deepeval_evaluation():
    samples_df = pd.read_csv(SAMPLE_PATH)
    generated_path = os.path.join(RESULT_DIR, "deepeval_generated_answers.csv")
    use_existing = os.getenv("USE_EXISTING_ANSWERS", "false").lower() == "true"

    if use_existing and os.path.exists(generated_path):
        logging.info(f"Loading existing answers from {generated_path}")
        generated_df = pd.read_csv(generated_path)
        generated_df['contexts'] = generated_df['contexts'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        eval_data = generated_df.to_dict('records')
    else:
        eval_data = asyncio.run(fetch_all_answers_in_batches(samples_df, batch_size=10))
        generated_df = pd.DataFrame(eval_data)
        generated_df.to_csv(generated_path, index=False)
        logging.info(f"Saved generated answers to {generated_path}")

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

    async_config = AsyncConfig(max_concurrent=3, throttle_value=2)
    evaluate(test_cases, metrics, async_config=async_config)


if __name__ == "__main__":
    run_deepeval_evaluation()
