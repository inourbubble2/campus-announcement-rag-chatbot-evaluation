# Campus Announcement RAG Chatbot Evaluation

Agentic RAG를 활용한 [챗봇](https://github.com/inourbubble2/campus-announcement-rag-chatbot)의 성능을 평가하기 위한 도구입니다. Ragas와 DeepEval 프레임워크를 모두 지원합니다.

## 기능

- Golden dataset을 사용한 자동화된 RAG 평가
- **Ragas Metrics**: Faithfulness, Answer Relevancy, Context Precision, Context Recall, Answer Correctness, URL Match
- **DeepEval Metrics**: Faithfulness, Answer Relevancy, Contextual Precision, Contextual Recall, GEval (Correctness)
- Conversation ID별 순차 처리로 대화 맥락 유지
- 배치 처리를 통한 효율적인 병렬 평가
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

| conversation_id | question | ground_truth | url |
|-----------------|----------|--------------|-----|
| 대화 세션 ID | 질문 내용 | 정답 내용 | 정답 URL |

**중요**: 같은 `conversation_id`를 가진 질문들은 순서대로 배치해야 합니다. 평가 시 같은 대화 내에서 순차적으로 처리되어 맥락이 유지됩니다.

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
  "contexts": ["컨텍스트1", "컨텍스트2", ...],
  "urls": ["https://example.com/notice?list_id=FA1&seq=29038", ...]
}
```

**대화 맥락 유지**: API는 `conversation_id`를 활용하여 이전 대화 내용을 기억하고 있어야 합니다.

3. 평가 실행

### Ragas 평가 실행

```bash
python evaluate_ragas.py
```

### DeepEval 평가 실행

```bash
python evaluate_deepeval.py
```

## 결과 확인

평가가 완료되면 `data/golden_results_YYYYMMDDHHMMSS.csv` 파일이 생성됩니다.

결과 파일에는 다음 정보가 포함됩니다:
- 각 질문에 대한 평가 결과
- 메트릭 점수 (0~1 범위)
  - **Ragas**: faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness, url_match
  - **DeepEval**: faithfulness, answer_relevancy, contextual_precision, contextual_recall, correctness (GEval)
- 예측된 URLs와 정답 URL
- 마지막 행에 각 메트릭의 평균 점수

### URL Match 평가 기준 (Ragas Only)

URL 일치도는 다음과 같이 평가됩니다:
- **1.0**: 예측 URL의 `list_id`와 `seq` 파라미터가 정답 URL과 모두 일치
- **0.0**: 일치하지 않음

예시:
```
정답: https://www.uos.ac.kr/korNotice/view.do?list_id=FA1&seq=29038&sort=16&...
예측: https://www.uos.ac.kr/korNotice/view.do?list_id=FA1&seq=29038&identified=anonymous
결과: 1.0 (같은 공지사항)
```

## 평가 모델

- LLM: `gpt-4o-mini`
- Embeddings: `text-embedding-3-small`

## 평가 프로세스

1. **데이터 로드**: `golden_samples.csv`에서 질문-답변 쌍 로드
2. **대화별 그룹핑**: `conversation_id`별로 질문 그룹핑
3. **순차/병렬 처리**:
   - 같은 대화 내 질문들: 순차 처리 (맥락 유지)
   - 다른 대화 간: 병렬 처리 (성능 최적화)
4. **RAGAS 평가**: 4개 메트릭 자동 평가
5. **URL 일치도 계산**: `list_id`와 `seq` 조합으로 URL 매칭
6. **결과 저장**: 타임스탬프가 포함된 CSV 파일로 저장

## 주요 설정

- `batch_size`: 동시 처리 conversation 수 (기본값: 10)
- 같은 대화 내 질문 간 대기 시간: 0.5초
