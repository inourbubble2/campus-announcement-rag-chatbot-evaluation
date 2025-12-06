import logging
import asyncio
import aiohttp
import os
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000/chat")

def calculate_url_match(predicted_urls, ground_truth_url):
  """
  예측된 URL들과 ground truth URL의 일치도를 계산
  list_id와 seq 파라미터 조합이 일치하면 같은 공지사항으로 판단

  Returns:
    float: 1.0 (list_id & seq 일치), 0.0 (일치 없음)
  """
  if not predicted_urls or not ground_truth_url:
    return 0.0

  parsed = urlparse(ground_truth_url)
  params = parse_qs(parsed.query)
  list_id = params.get('list_id', [None])[0]
  seq = params.get('seq', [None])[0]

  for url in predicted_urls:
    predicted_parsed = urlparse(url)
    predicted__params = parse_qs(predicted_parsed.query)
    predicted_list_id = predicted__params.get('list_id', [None])[0]
    predicted_seq = predicted__params.get('seq', [None])[0]

    if predicted_list_id == list_id and predicted_seq == seq:
      return 1.0

  return 0.0


async def call_chat_api_async(session, index, question, conversation_id, ground_truth, ground_truth_url):
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
      urls = data.get("urls", [])

      logging.info(f"Processed A{index+1}: {answer.replace(chr(10), ' ')[:100]}...")

      return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth,
        "ground_truth_url": ground_truth_url,
        "predicted_urls": urls,
      }
  except Exception as e:
    logging.error(f"API error for Q{index+1}: {e}")
    return None


async def process_conversation_sequentially(session, conversation_group):
  """같은 conversation_id를 가진 질문들을 순차적으로 처리"""
  results = []

  for index, row in conversation_group.iterrows():
    result = await call_chat_api_async(
      session,
      index,
      row['question'],
      row['conversation_id'],
      row['ground_truth'],
      row['url']
    )
    if result is not None:
      results.append(result)

    # 같은 대화 내에서 다음 질문 전 짧은 대기
    await asyncio.sleep(0.5)

  return results


async def fetch_all_answers_in_batches(samples_df, batch_size=10):
  """
  conversation_id별로 그룹핑하여:
  - 같은 conversation_id 내에서는 순차 처리 (맥락 유지)
  - 다른 conversation_id는 병렬 처리 (속도 향상)
  """
  eval_data = []

  # conversation_id별로 그룹핑
  grouped = samples_df.groupby('conversation_id', sort=False)
  conversation_groups = [group for _, group in grouped]

  logging.info(f"Total conversations: {len(conversation_groups)}")

  async with aiohttp.ClientSession() as session:
    # conversation 단위로 배치 처리
    for i in range(0, len(conversation_groups), batch_size):
      batch = conversation_groups[i:i+batch_size]
      logging.info(f"Processing conversation batch {i//batch_size + 1} "
                   f"({i+1}-{min(i+batch_size, len(conversation_groups))} "
                   f"of {len(conversation_groups)} conversations)")

      # 각 conversation은 내부적으로 순차 처리, 다른 conversation과는 병렬 처리
      tasks = [process_conversation_sequentially(session, conv_group)
               for conv_group in batch]

      batch_results = await asyncio.gather(*tasks)

      # 결과 평탄화
      for conversation_results in batch_results:
        eval_data.extend(conversation_results)

  return eval_data
