import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# pipeline의 모듈들을 가져옵니다.
from pipeline.indexer import tokenize_ko
from pipeline.embedder import Embedder
from pipeline.reranker import Reranker

# 환경 변수 로드를 위해 dotenv를 사용합니다. (실제 운영에서는 컨테이너 환경변수 사용)
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# 모델 로딩 (애플리케이션 시작 시)
embedder = Embedder()
reranker = Reranker()

# --- Pydantic 모델 정의 ---
class IndexRequest(BaseModel):
    title: str
    body: str
    url: str

class AskRequest(BaseModel):
    query: str

# --- API 엔드포인트 ---

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/index")
def index_document(doc: IndexRequest):
    # 1. 텍스트 청킹 (간단하게 body를 그대로 사용)
    chunk = doc.body

    # 2. 한국어 토큰화 (MeiliSearch용)
    tokens = tokenize_ko(chunk)

    # 3. 벡터 임베딩 (Qdrant용)
    vector = embedder.encode([chunk])[0]

    # 이 예제에서는 실제 DB/검색엔진에 저장하는 대신, 결과만 출력합니다.
    print(f"Indexing new document:")
    print(f"  - Title: {doc.title}")
    print(f"  - Tokens (for MeiliSearch): {tokens}")
    print(f"  - Vector (for Qdrant): {len(vector)} dims")

    # 실제 구현에서는 아래와 같은 작업이 필요합니다.
    # meili_client.index(index_name).add_documents([{"id": ..., "title": ..., "tokens": ...}])
    # qdrant_client.upsert(collection_name, points=[PointStruct(id=..., vector=..., payload=...)])

    return {"message": "Document received for indexing", "title": doc.title}

@app.post("/ask")
def ask_question(req: AskRequest):
    query = req.query
    print(f"Received query: {query}")

    # 1. 하이브리드 검색 (MeiliSearch BM25 + Qdrant Vector Search)
    # 이 단계는 실제 검색엔진 연동이 필요하므로, 더미 데이터를 생성합니다.
    candidates = [
        {"text": "반품은 수령일로부터 14일 이내 가능합니다.", "score": 0.85},
        {"text": "기술 지원은 홈페이지를 통해 접수해주세요.", "score": 0.79},
        {"text": "배송 조회는 마이페이지에서 확인 가능합니다.", "score": 0.65},
    ]

    # 2. 조건부 재랭킹
    CONF_TH = float(os.getenv("CONFIDENCE_TH", "0.52"))
    RERANK_COND_GAP = float(os.getenv("RERANK_COND_GAP", "0.06"))
    
    scores = [c["score"] for c in candidates]
    need_rerank = (max(scores) < CONF_TH) or ((scores[0] - scores[1]) < RERANK_COND_GAP)
    
    if need_rerank:
        print("Condition met. Performing reranking...")
        reranked_results = reranker.rerank(query, [c["text"] for c in candidates])
        # 재정렬된 결과에 따라 원래 후보 순서를 맞춥니다.
        final_candidates = [candidates[res["idx"]] for res in reranked_results]
        final_candidates[0]["score"] = reranked_results[0]["score"] # 재랭킹된 점수로 업데이트
    else:
        print("Condition not met. Skipping reranking.")
        final_candidates = candidates

    # 3. 최종 답변 생성
    best_answer = final_candidates[0]

    if best_answer["score"] < CONF_TH:
        return {"answer": "죄송합니다. 질문에 대한 정확한 답변을 찾기 어렵습니다. 추가 정보를 제공해주시겠어요?", "score": best_answer["score"]}
    else:
        return {"answer": best_answer["text"], "score": best_answer["score"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
