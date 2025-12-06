import pandas as pd
import os
import logging
import asyncio
from datetime import datetime
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
  faithfulness,
  answer_relevancy,
  context_precision,
  context_recall,
  answer_correctness,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from utils import (
    calculate_url_match,
    fetch_all_answers_in_batches
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

API_URL = os.getenv("API_URL", "http://localhost:8000/chat")
SAMPLE_PATH = os.getenv("SAMPLE_PATH", "data/golden_samples.csv")
RESULT_DIR = os.getenv("RESULT_DIR", "data")

llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def run_ragas_evaluation():
  """Golden dataset을 사용하여 RAGAS 평가를 실행"""
  timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
  result_path = os.path.join(RESULT_DIR, f"golden_results_{timestamp}.csv")

  samples_df = pd.read_csv(SAMPLE_PATH)
  logging.info(f"Loaded {len(samples_df)} samples from {SAMPLE_PATH}")

  eval_data = asyncio.run(fetch_all_answers_in_batches(samples_df, batch_size=10))

  # 원본 데이터
  eval_df = pd.DataFrame(eval_data)

  # RAGAS evaluation을 위한 dataset
  dataset = Dataset.from_pandas(eval_df[['question', 'answer', 'contexts', 'ground_truth']])

  metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
  ]

  logging.info("Running RAGAs evaluation with Ground Truth...")
  results = evaluate(
      dataset=dataset,
      metrics=metrics,
      llm=llm,
      embeddings=embeddings,
  )

  logging.info("=== Evaluation Results ===")
  logging.info(results)

  evaluation_results_df = results.to_pandas()

  # 원본 데이터의 URL 정보를 evaluation 결과에 추가
  evaluation_results_df['ground_truth_url'] = eval_df['ground_truth_url'].values
  evaluation_results_df['predicted_urls'] = eval_df['predicted_urls'].values

  # URL 일치도 계산
  evaluation_results_df['url_match'] = evaluation_results_df.apply(
    lambda row: calculate_url_match(
      row.get('predicted_urls', []),
      row.get('ground_truth_url', '')
    ),
    axis=1
  )

  metric_columns = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'answer_correctness', 'url_match']
  avg_values = evaluation_results_df[metric_columns].mean()

  logging.info(f"=== URL Match Score: {avg_values['url_match']:.4f} ===")

  avg_row = pd.DataFrame({
    'user_input': [''],
    'response': [''],
    'retrieved_contexts': [''],
    'reference': ['AVERAGE'],
    'faithfulness': [avg_values['faithfulness']],
    'answer_relevancy': [avg_values['answer_relevancy']],
    'context_precision': [avg_values['context_precision']],
    'context_recall': [avg_values['context_recall']],
    'answer_correctness': [avg_values['answer_correctness']],
    'url_match': [avg_values['url_match']]
  })

  evaluation_results_df = pd.concat([evaluation_results_df, avg_row], ignore_index=True)
  evaluation_results_df.to_csv(result_path, index=False)

  logging.info(f"Detailed results saved to {result_path}")


if __name__ == "__main__":
  run_ragas_evaluation()
