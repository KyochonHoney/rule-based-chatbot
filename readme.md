# 고성능 하이브리드 검색 챗봇 (ONNX 최적화)

본 프로젝트는 GPU 없이 일반 CPU 환경에서 높은 성능을 내는 하이브리드 검색(Hybrid Search) 챗봇 엔진입니다. 문서의 내용을 기반으로 질문에 가장 정확한 답변을 찾아 제공하며, 특히 한국어 처리에 최적화되어 있습니다.

## ✨ 주요 특징

- **하이브리드 검색**: 키워드 기반 검색(BM25)과 의미 기반 벡터 검색(Dense)을 결합하여 검색 정확도를 극대화합니다.
- **조건부 재랭킹 (Conditional Reranking)**: 1차 검색 결과의 신뢰도가 낮을 경우에만 경량 재랭커 모델을 사용하여, 불필요한 연산을 줄이고 평균 응답 속도를 향상시킵니다.
- **ONNX 런타임 기반 고성능 추론**: PyTorch 모델을 INT8 양자화된 ONNX 모델로 변환하여 CPU 환경에서 30~60% 향상된 추론 속도를 제공합니다.
- **한국어 처리 최적화**: `kiwipiepy` 형태소 분석기를 사용하여 조사를 포함한 다양한 한국어 질의에 대한 검색 성능을 높였습니다.
- **Docker 기반 실행 환경**: Docker를 통해 복잡한 설정 없이 일관된 환경에서 간편하게 서비스를 실행할 수 있습니다.

## ⚙️ 아키텍처 개요

```
[Client]
   │ HTTP
[FastAPI App (Docker)]  →  Redis(캐시/큐) - (미구현)
         ├→ Meilisearch(BM25; tokens,title) - (미구현)
         ├→ Qdrant(벡터; HNSW) - (미구현)
         └→ PostgreSQL(메타/로그) - (미구현)

[ONNX Models]
  - Embedding (multilingual-e5-small)
  - Reranker (ms-marco-MiniLM-L-6-v2)
```

> 현재 구현은 FastAPI 애플리케이션과 ONNX 모델 추론 로직에 집중되어 있으며, 데이터베이스 및 외부 검색 엔진 연동은 향후 확장 가능하도록 구조화되어 있습니다.

## 📂 프로젝트 구조

```
.
├── app/                  # FastAPI 핵심 애플리케이션
│   ├── main.py           # API 엔드포인트 정의
│   └── requirements.txt  # Python 의존성 목록
├── pipeline/             # 데이터 처리 파이프라인
│   ├── embedder.py       # 임베딩 모델 로더
│   ├── indexer.py        # 한국어 토크나이저
│   └── reranker.py       # 재랭킹 모델 로더
├── scripts/              # (참고용) 원본 모델 변환 스크립트
├── .env.example          # 환경변수 예시 파일
├── .gitignore            # Git 무시 파일 목록
├── Dockerfile            # 서비스 Docker 이미지 빌드 파일
├── docker-compose.yml    # Docker 서비스 실행 및 관리 파일
└── prepare_models.py     # 모델 다운로드 및 ONNX 변환 스크립트
```

## 🚀 시작하기

### 사전 요구사항

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) 설치
- [Python 3.10+](https://www.python.org/downloads/) 설치

### 1. 모델 준비

챗봇에 필요한 임베딩/재랭커 모델을 다운로드하고 ONNX 형식으로 변환합니다. 프로젝트 루트 디렉터리에서 아래 명령어를 실행하세요. (필요한 라이브러리는 스크립트가 자동으로 설치합니다.)

```bash
python prepare_models.py
```

이 스크립트는 `C:\opt\models` 디렉터리를 생성하고 그 안에 `emb` 및 `rerank` 모델을 저장합니다.

### 2. 환경 변수 설정

`.env.example` 파일을 복사하여 `.env` 파일을 생성합니다. 모델 경로는 기본적으로 설정되어 있으므로 별도 수정은 필요하지 않습니다.

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

### 3. 챗봇 서비스 실행

Docker Desktop이 실행 중인 상태에서 아래 명령어를 실행하여 챗봇 서비스를 빌드하고 시작합니다.

```bash
docker compose up -d --build
```

### 4. 서비스 상태 확인

서비스가 정상적으로 실행되었는지 확인합니다.

```bash
curl http://localhost:8000/health
```

`{"status":"ok"}` 와 같은 응답이 오면 성공입니다.

## 📖 API 사용법

### 문서 색인 (Indexing)

챗봇이 답변할 지식 문서를 추가합니다. (현재는 DB 연동 없이 로그만 출력됩니다.)

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d 
  {
    "title": "반품 정책",
    "body": "반품은 수령일로부터 14일 이내 가능합니다. 영수증 지참 필요...",
    "url": "https://example.com/returns"
  }
```

### 질문하기 (Ask)

챗봇에게 질문을 합니다.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "반품 기간이 어떻게 되나요?"}'
```

**응답 예시:**
```json
{
  "answer": "반품은 수령일로부터 14일 이내 가능합니다.",
  "score": 0.85
}
```