# Campus Noti RAGAS Evaluation

RAG(Retrieval-Augmented Generation) 챗봇의 성능을 평가하기 위한 RAGAS 기반 평가 도구입니다.

## 기능

- Golden dataset을 사용한 자동화된 RAG 평가
- RAGAS 메트릭을 통한 다차원 성능 측정
  - **Faithfulness**: 답변이 제공된 컨텍스트에 얼마나 충실한지
  - **Answer Relevancy**: 답변이 질문과 얼마나 관련있는지
  - **Context Precision**: 검색된 컨텍스트의 정확도
  - **Context Recall**: Ground truth 대비 컨텍스트 재현율
- CSV 형식으로 평가 결과 저장 및 평균 점수 계산

## 요구사항

- Python 3.10+
- OpenAI API Key

## 설치

1. 가상환경 생성 및 활성화
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2. 의존성 설치
```bash
pip install -r requirements.txt
```

## 환경 설정

`.env` 파일을 생성하고 다음 환경변수를 설정하세요:

```env
OPENAI_API_KEY=your_openai_api_key_here
API_URL=http://localhost:8000/chat
SAMPLE_PATH=data/golden_samples.csv
RESULT_PATH=data/golden_results.csv
```

## 사용 방법

1. Golden dataset 준비

`data/golden_samples.csv` 파일을 다음 형식으로 준비하세요:

| question | ground_truth | conversation_id |
|----------|--------------|-----------------|
| 질문 내용 | 정답 내용 | 대화 세션 ID |

2. 챗봇 API 서버 실행

평가할 챗봇 API가 `API_URL`에 지정된 주소에서 실행 중이어야 합니다.

API는 다음 형식의 요청을 받아야 합니다:
```json
{
  "question": "질문 내용",
  "conversation_id": "대화 ID"
}
```

그리고 다음 형식으로 응답해야 합니다:
```json
{
  "answer": "답변 내용",
  "contexts": ["컨텍스트1", "컨텍스트2", ...]
}
```

3. 평가 실행

```bash
python evaluate.py
```

## 결과 확인

평가가 완료되면 `data/golden_results.csv` 파일이 생성됩니다.

결과 파일에는 다음 정보가 포함됩니다:
- 각 질문에 대한 평가 결과
- 4가지 메트릭 점수 (0~1 범위)
- 마지막 행에 각 메트릭의 평균 점수

## 평가 모델

- LLM: `gpt-4o-mini`
- Embeddings: `text-embedding-3-small`
