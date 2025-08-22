# 단일 VPC 1대 챗봇 엔진 구축 안내서 (한국어 최적화 적용)

본 문서는 GPU 없이 **일반 VPC 1대**에서 실행되는 하이브리드 검색형 챗봇 엔진의 **구축 절차**를 제공합니다. 기존 all‑in‑one 번들(FastAPI App, Worker, Qdrant, Meilisearch, Redis, PostgreSQL, Nginx)을 기반으로 **한국어 형태소(kiwi) 사전 토큰화**와 **조건부 재랭킹(신뢰도 기반)**을 적용합니다.

---

## 1. 아키텍처 개요
```
[Client]
   │ HTTPS
[Nginx]  →  [App(FastAPI)]  →  Redis(캐시/큐)
                     ├→ Meilisearch(BM25; tokens,title)
                     ├→ Qdrant(벡터; HNSW)
                     └→ PostgreSQL(메타/로그)
[Worker] ←(수집/색인/임베딩 배치)
```
- **하이브리드 검색**: BM25(Meili) + 코사인(Qdrant) 점수 융합 → **신뢰도 임계값 미달 시** 경량 Cross‑Encoder 재랭킹/추가질문.
- **한국어 최적화**: 색인 시 **kiwipiepy(kiwi)** 로 형태소 분해 → Meili의 `tokens` 필드에 공백 구분 토큰 저장.

---

## 2. 요구 사양(권장)
- VPC 1대: **8 vCPU / 16~32GB RAM / 200GB SSD**, Ubuntu 22.04 LTS
- 공개 포트: 80(또는 443), 7700(Meili), 6333(Qdrant), 6379(Redis), 5432(Postgres) — 운영 시 외부 노출 최소화(보안그룹 제한)

---

## 3. 사전 준비
1) OS 업데이트 및 필수 패키지 설치
```bash
sudo apt update && sudo apt -y upgrade
sudo apt -y install ca-certificates curl gnupg build-essential ufw
```
2) Docker & Compose 설치
```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
# (재로그인 필요)
```
3) 방화벽/보안그룹(예시: 80/443만 외부 허용, 나머지는 로컬/내부망)
```bash
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw enable
```

---

## 4. 소스 준비 및 환경 설정
1) 저장소 준비(프로젝트 루트: `all-in-one/` 가정)
```bash
# 사내 Git 또는 로컬 복사본 기준
cd /opt && sudo mkdir chatbot && sudo chown $USER:$USER chatbot && cd chatbot
# (여기에 all-in-one 디렉터리 배치)
```
2) 환경 변수 파일
```bash
cp .env.example .env
# 필요 시 DB 비밀번호, 임계값, TOPK 조정
```

---

## 5. 한국어 형태소(kiwi) 사전 토큰화 적용
### 5.1 의존성 추가
- `app/requirements.txt`에 `kiwipiepy==0.21.0` 추가.

### 5.2 Meili 인덱스 설정(토큰 필드 사용)
부팅 후 1회 설정:
```bash
# searchableAttributes: tokens 우선, 그 다음 title
curl -X PATCH "$MEILI_HOST/indexes/$MEILI_INDEX/settings" \
  -H 'Content-Type: application/json' \
  -d '{
    "searchableAttributes": ["tokens", "title"],
    "displayedAttributes": ["title", "text", "tokens"],
    "typoTolerance": {"enabled": true}
  }'
```

### 5.3 색인 파이프라인 수정(pipeline/indexer.py)
- 문단 청킹 후, 각 청크를 **kiwi**로 형태소 분해하여 `tokens` 문자열 생성.
- `/index` API에서 Meili 문서에 `tokens` 필드를 포함.

예시 스니펫:
```python
# pipeline/indexer.py (예시)
from kiwipiepy import Kiwi
kiwi = Kiwi()

def tokenize_ko(text: str) -> str:
    # 형태소 표층형 중 명사/동사어간/형용사어간 위주 추출(간단 예시)
    toks = []
    for sent in kiwi.tokenize(text):
        for t in sent:
            pos = t.tag.split("/")[0]
            if pos in ("NNG","NNP","VV","VA","MAG","SL"):
                toks.append(t.form)
    return " ".join(toks)
```
`app/main.py`의 `/index` 구현에서:
```python
# 생성되는 ms_docs 항목에 tokens 추가
ms_docs.append({
  "id": h, "doc_id": doc_id, "title": doc.title,
  "text": ch, "tokens": tokenize_ko(ch)
})
```

> **효과**: 한국어 조사/어미 변화에 강해져 BM25 리콜 향상.

---

## 6. 조건부 재랭킹과 신뢰도 정책
### 6.1 환경 변수
`.env`에서 (예시)
```
CONFIDENCE_TH=0.52     # 신뢰도 임계값
RERANK_COND_GAP=0.06    # 1,2위 점수 격차가 이 값보다 작으면 재랭킹 수행
TOPK=8                  # 1차 하이브리드 후보 수
```

### 6.2 로직 개요(app/main.py의 /ask)
1) 하이브리드 상위 k 후보 스코어 집합 계산.
2) **최고 점수 < 임계값** 또는 **상위 1·2위 격차 < RERANK_COND_GAP** 이면 **재랭킹 실행**.
3) 재랭킹 후에도 임계값 미달 시 **추가질문/사람연결**로 폴백.

예시 스니펫:
```python
scores = [c["score"] for c in candidates]
need_rerank = (max(scores) < CONF_TH) or ((scores[0]-scores[1]) < float(os.getenv("RERANK_COND_GAP","0.06")))
if need_rerank:
    reranked = reranker.rerank(q, [c["text"] for c in candidates])
else:
    reranked = [{"idx":0, "score":scores[0]}]
```

### 6.3 재랭커 모델 권장
- 비용·품질 균형: **다국어 MiniLM/BAAI bge‑reranker‑base**(INT8 ONNX 권장)
- 적용 방식: Top‑k(20 이하)만 쿼리·문단 쌍 점수화 → 최고 점수 기준 응답.

---

## 7. 기동
```bash
# 최초 기동
docker compose up -d --build
# 상태 확인
curl http://<서버IP>/health
```

---

## 8. 초기 색인 및 조회 테스트
### 8.1 색인
```bash
curl -X POST http://<서버IP>/index \
  -H 'Content-Type: application/json' \
  -d '{
    "title": "반품 정책",
    "body": "반품은 수령일로부터 14일 이내 가능합니다. 영수증 지참 필요...",
    "url": "https://example.com/returns"
  }'
```

### 8.2 질의
```bash
curl -X POST http://<서버IP>/ask \
  -H 'Content-Type: application/json' \
  -d '{"query": "반품 기간이 어떻게 되나요?"}'
```

---

## 9. 성능 목표 및 튜닝 가이드(8vCPU/16–32GB)
- **BM25 전용** 300–600 QPS, **하이브리드** 150–300 QPS, **재랭킹 포함** 40–80 QPS(보수적)
- **캐시 히트 30%**만 확보해도 체감 QPS 1.3~1.8배 ↑
- Qdrant: `efSearch` 하향 → 지연↓/QPS↑(정확도 약간 감소)
- Top‑k 8→6 조정, 재랭킹 조건부 실행으로 평균 지연 감소
- 임베딩/재랭킹 **ONNX INT8** 사용(30~60% CPU 절감)

---

## 10. 운영(보안/모니터링/백업)
- **보안**: Nginx RateLimit, API 키, 443/TLS(ACME), 관리자 대역 화이트리스트
- **모니터링**: Prometheus+Grafana(컨테이너 리소스/HTTP 레이턴시), EFK(로그), 알람(Webhook)
- **백업**:
  - PostgreSQL: `pg_dump` 일일 스냅샷
  - Qdrant: 스냅샷 기능 사용(데이터 디렉터리 보존)
  - Meili: `meili dumps` 주기 실행
- **DR**: 볼륨 주기 스냅샷(클라우드 블록 스토리지 스냅샷 권장)

---

## 11. 수용 가능 동시사용자(참고)
- 평균 사고시간 20초 가정 시: **BM25 300 QPS ≈ 6,000 동시**, **하이브리드 200 QPS ≈ 4,000 동시**, **재랭크 60 QPS ≈ 1,200 동시**
- FAQ/핫쿼리 캐시 비율이 높을수록 실효 동시사용자 ↑

---

## 12. 확장 및 대안
- **OpenSearch(+Nori)**: 한국어 BM25 정밀도↑(형태소/어간/복합명사), 단일 노드 메모리 여유 필요
- **pgvector 단순화**: Qdrant 제거, 규모가 작을 때 운영 간소화 가능
- **샤딩/분리**: 트래픽 증가 시 App/Worker, Meili/Qdrant, DB를 순차 분리

---

## 13. 점검 체크리스트(Go‑Live 전)
- [ ] `/health` 200 OK
- [ ] Meili 인덱스 settings에 `searchableAttributes = ["tokens","title"]` 반영
- [ ] 색인 문서에 `tokens` 필드 생성 확인
- [ ] Qdrant 컬렉션 생성 및 검색 동작 확인
- [ ] 캐시 TTL 동작, RateLimit 적용
- [ ] 재랭킹 조건부 트리거 테스트(임계값 하향으로 강제)
- [ ] 백업/복구 리허설 1회 완료
- [ ] 보안그룹/방화벽 외부 노출 점검(80/443 외 차단)

---

## 14. 부록: 구성 파일 변경 요약
- `app/requirements.txt`: `kiwipiepy` 추가
- `pipeline/indexer.py`: `tokenize_ko()` 도입, `/index` 경로에서 `tokens` 포함
- `app/main.py`: 조건부 재랭킹 로직, 신뢰도/격차 환경 변수 반영
- Meili 인덱스 설정: `searchableAttributes`에 `tokens` 포함(초기 1회 패치)

---

