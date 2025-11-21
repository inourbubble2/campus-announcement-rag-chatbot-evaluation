import pandas as pd
import os
import logging
import asyncio
import aiohttp
from datetime import datetime
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
  faithfulness,
  answer_relevancy,
  context_precision,
  context_recall,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

API_URL = os.getenv("API_URL", "http://localhost:8000/chat")
SAMPLE_PATH = os.getenv("SAMPLE_PATH", "data/golden_samples.csv")
RESULT_DIR = os.getenv("RESULT_DIR", "data")

llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


async def call_chat_api_async(session, index, question, conversation_id, ground_truth):
  logging.info(f"Processing Q{index+1}: {question[:50]}...")

  try:
    async with session.post(
        API_URL,
        json={"question": question, "conversation_id": conversation_id},
        headers={"Content-Type": "application/json"}
    ) as response:
      response.raise_for_status()
      data = await response.json()

      answer = data.get("answer", "")
      contexts = data.get("contexts", [])

      logging.info(f"Processed A{index+1}: {answer.replace(chr(10), ' ')[:100]}...")

      return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth
      }
  except Exception as e:
    logging.error(f"API error for Q{index+1}: {e}")
    return None


async def fetch_all_answers_in_batches(samples_df, batch_size=10):
  """Golden dataset의 모든 질문에 대해 배치 단위로 답변을 병렬 수집"""
  eval_data = []

  async with aiohttp.ClientSession() as session:
    for i in range(0, len(samples_df), batch_size):
      batch = samples_df.iloc[i:i+batch_size]
      logging.info(f"Processing batch {i//batch_size + 1} ({i+1}-{min(i+batch_size, len(samples_df))} of {len(samples_df)})")

      tasks = []
      for index, row in batch.iterrows():
        task = call_chat_api_async(
          session,
          index,
          row['question'],
          row['conversation_id'],
          row['ground_truth']
        )
        tasks.append(task)

      results = await asyncio.gather(*tasks)

      for result in results:
        if result is not None:
          eval_data.append(result)

  return eval_data


def run_ragas_evaluation():
  """Golden dataset을 사용하여 RAGAS 평가를 실행"""
  timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
  result_path = os.path.join(RESULT_DIR, f"golden_results_{timestamp}.csv")

  samples_df = pd.read_csv(SAMPLE_PATH)
  logging.info(f"Loaded {len(samples_df)} samples from {SAMPLE_PATH}")

  eval_data = asyncio.run(fetch_all_answers_in_batches(samples_df, batch_size=10))

  dataset = Dataset.from_pandas(pd.DataFrame(eval_data))

  metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
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

  metric_columns = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
  avg_values = evaluation_results_df[metric_columns].mean()

  avg_row = pd.DataFrame({
    'user_input': [''],
    'retrieved_contexts': [''],
    'response': [''],
    'reference': ['AVERAGE'],
    'faithfulness': [avg_values['faithfulness']],
    'answer_relevancy': [avg_values['answer_relevancy']],
    'context_precision': [avg_values['context_precision']],
    'context_recall': [avg_values['context_recall']]
  })

  evaluation_results_df = pd.concat([evaluation_results_df, avg_row], ignore_index=True)
  evaluation_results_df.to_csv(result_path, index=False)

  logging.info(f"Detailed results saved to {result_path}")


if __name__ == "__main__":
  run_ragas_evaluation()
