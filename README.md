# Campus Announcement RAG Chatbot Evaluation

Agentic RAG를 활용한 [챗봇](https://github.com/inourbubble2/campus-announcement-rag-chatbot)의 성능을 평가하기 위한 도구입니다.

## 기능

- Golden dataset을 사용한 자동화된 RAG 평가
- **Ragas Metrics**: Faithfulness, Answer Relevancy, Context Precision, Context Recall, Answer Correctness
- Conversation ID별 순차 처리로 대화 맥락 유지
- 배치 처리를 통한 병렬 평가
- CSV 형식으로 평가 결과 저장 및 평균 점수 계산
