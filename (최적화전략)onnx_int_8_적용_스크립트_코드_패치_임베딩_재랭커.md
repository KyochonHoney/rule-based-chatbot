# ONNX INT8 적용 스크립트 & 코드 패치 (임베딩/재랭커)

단일 VPC 1대 all‑in‑one 번들에 **임베딩/재랭커 ONNX(INT8)**를 적용하기 위한 스크립트와 코드 패치입니다.

- 기본 임베딩 모델(권장): `intfloat/multilingual-e5-small` *(다국어 성능·속도 균형)*
- 대안 임베딩: `sentence-transformers/distiluse-base-multilingual-cased-v2`
- 기본 재랭커(권장): `jinaai/jina-reranker-v2-base-multilingual`
- 대안 재랭커: `cross-encoder/ms-marco-MiniLM-L-6-v2` *(영어 중심)*

> 모델명은 모두 환경변수로 바꿀 수 있습니다.

---

## 0) 디렉터리 & 요구 패키지 업데이트
```
all-in-one/
├─ scripts/
│  ├─ export_onnx.sh           # 모델 다운로드 + ONNX 변환 (optimum-cli)
│  ├─ quantize_onnx.py         # ONNX → INT8 동적 양자화(onnxruntime)
│  └─ verify_onnx.py           # 간단 검증 스크립트(선택)
└─ app/requirements.txt        # transformers/optimum/onnxruntime 추가
```

### app/requirements.txt (추가)
```diff
 fastapi==0.115.0
 uvicorn[standard]==0.30.6
 httpx==0.27.0
 redis==5.0.7
 psycopg2-binary==2.9.9
 meilisearch==0.32.2
 qdrant-client==1.9.1
 numpy==1.26.4
 scikit-learn==1.5.1
-onnxruntime==1.18.0
-sentence-transformers==3.0.1
+onnxruntime==1.18.0
+sentence-transformers==3.0.1
+transformers==4.43.3
+optimum==1.21.4
+tokenizers==0.19.1
 python-dotenv==1.0.1
 kiwipiepy==0.21.0
```

> 이미지 재빌드 필요: `docker compose build app worker`

---

## 1) ONNX 내보내기 스크립트 (scripts/export_onnx.sh)
```bash
#!/usr/bin/env bash
set -euo pipefail

# ===== 사용자 설정 =====
EMBED_MODEL=${EMBED_MODEL:-"intfloat/multilingual-e5-small"}
RERANK_MODEL=${RERANK_MODEL:-"jinaai/jina-reranker-v2-base-multilingual"}
OUT_DIR=${OUT_DIR:-"/opt/models"}

mkdir -p "$OUT_DIR/emb" "$OUT_DIR/rerank"

# 1) 임베딩 모델 → ONNX
python -m pip install --no-cache-dir --upgrade "optimum>=1.20.0" "transformers>=4.40.0" tokenizers

echo "[export] embedding model → ONNX: $EMBED_MODEL"
optimum-cli export onnx \
  --model "$EMBED_MODEL" \
  --task feature-extraction \
  "$OUT_DIR/emb"

# 2) 재랭커 모델 → ONNX (sequence-classification)
echo "[export] reranker model → ONNX: $RERANK_MODEL"
optimum-cli export onnx \
  --model "$RERANK_MODEL" \
  --task text-classification \
  "$OUT_DIR/rerank"

# 토크나이저/구성은 각각 디렉터리에 자동 저장됨
ls -al "$OUT_DIR/emb" || true
ls -al "$OUT_DIR/rerank" || true

cat <<EOF
[ok] Export finished.
Embedding ONNX dir: $OUT_DIR/emb
Reranker ONNX dir:  $OUT_DIR/rerank
EOF
```

실행 예시:
```bash
sudo bash scripts/export_onnx.sh \
  EMBED_MODEL=intfloat/multilingual-e5-small \
  RERANK_MODEL=jinaai/jina-reranker-v2-base-multilingual \
  OUT_DIR=/opt/models
```

---

## 2) INT8 양자화 스크립트 (scripts/quantize_onnx.py)
```python
#!/usr/bin/env python
import argparse
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

parser = argparse.ArgumentParser()
parser.add_argument("--in", dest="in_path", required=True, help="input ONNX model path (.onnx)")
parser.add_argument("--out", dest="out_path", required=True, help="output INT8 ONNX path (.onnx)")
parser.add_argument("--per_channel", action="store_true", help="enable per-channel quant")
args = parser.parse_args()

inp = Path(args.in_path)
out = Path(args.out_path)
out.parent.mkdir(parents=True, exist_ok=True)

print(f"[quantize] {inp} → {out} (INT8 dynamic, per_channel={args.per_channel})")
quantize_dynamic(
    model_input=str(inp),
    model_output=str(out),
    weight_type=QuantType.QInt8,
    per_channel=args.per_channel,
    reduce_range=False,
)
print("[ok] quantized")
```

실행 예시:
```bash
# 임베딩 ONNX 양자화
python3 scripts/quantize_onnx.py \
  --in /opt/models/emb/model.onnx \
  --out /opt/models/emb/model.int8.onnx \
  --per_channel

# 재랭커 ONNX 양자화
python3 scripts/quantize_onnx.py \
  --in /opt/models/rerank/model.onnx \
  --out /opt/models/rerank/model.int8.onnx \
  --per_channel
```

---

## 3) 간단 검증 스크립트 (scripts/verify_onnx.py)
```python
#!/usr/bin/env python
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import sys

mode = sys.argv[1]  # emb | rerank
model_dir = sys.argv[2]
text = sys.argv[3] if len(sys.argv)>3 else "반품 정책은 어떻게 되나요?"

sess = ort.InferenceSession(f"{model_dir}/model.int8.onnx", providers=["CPUExecutionProvider"])
tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

if mode == 'emb':
    batch = tok([text], padding=True, truncation=True, return_tensors='np', max_length=512)
    outputs = sess.run(None, {k: batch[k] for k in ["input_ids","attention_mask"] if k in batch})
    last = outputs[0]  # (B, T, H) or (B, H) depending on exporter
    if last.ndim == 3:
        mask = batch["attention_mask"][..., None]
        sum_vec = (last * mask).sum(axis=1)
        denom = np.clip(mask.sum(axis=1), 1e-9, None)
        pooled = sum_vec / denom
    else:
        pooled = last
    # L2 normalize
    norm = np.linalg.norm(pooled, axis=1, keepdims=True)
    pooled = pooled / np.clip(norm, 1e-9, None)
    print("emb dim:", pooled.shape)

elif mode == 'rerank':
    # pair 입력
    query = text
    doc = "반품은 배송 후 14일 이내 가능합니다."
    batch = tok([query], [doc], padding=True, truncation=True, return_tensors='np', max_length=512)
    outputs = sess.run(None, {k: batch[k] for k in ["input_ids","attention_mask","token_type_ids"] if k in batch})
    logits = outputs[0]
    print("logits:", logits)
else:
    raise SystemExit("mode must be emb|rerank")
```

---

## 4) 애플리케이션 코드 패치
### 4.1 pipeline/embedder.py (ONNX 경로 사용 + 토크나이저, 평균풀링)
```python
import os
import numpy as np
from typing import List
import onnxruntime as ort
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self):
        self.onnx_dir = os.getenv("EMBED_ONNX_DIR", os.getenv("EMBED_ONNX_PATH", "")).rstrip("/")
        self.model_name = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")

        if self.onnx_dir and os.path.exists(os.path.join(self.onnx_dir, "model.int8.onnx")):
            self.sess = ort.InferenceSession(os.path.join(self.onnx_dir, "model.int8.onnx"), providers=["CPUExecutionProvider"])
            self.tok = AutoTokenizer.from_pretrained(self.onnx_dir, use_fast=True)
            # 추정 차원: 첫 추론 후 세팅
            self.dim = None
            self.st_model = None
        else:
            # fallback: PyTorch ST (개발/검증용)
            self.sess = None
            self.tok = None
            self.st_model = SentenceTransformer(self.model_name)
            self.dim = self.st_model.get_sentence_embedding_dimension()

    def _mean_pool(self, last_hidden, attn_mask):
        mask = np.expand_dims(attn_mask, axis=-1)
        summed = (last_hidden * mask).sum(axis=1)
        denom = np.clip(mask.sum(axis=1), 1e-9, None)
        return summed / denom

    def encode(self, texts: List[str]):
        if self.sess is None:
            vecs = self.st_model.encode(texts, normalize_embeddings=True)
            if self.dim is None:
                self.dim = vecs.shape[-1]
            return vecs.tolist()

        batch = self.tok(texts, padding=True, truncation=True, return_tensors='np', max_length=512)
        feeds = {k: batch[k] for k in ["input_ids","attention_mask"] if k in batch}
        outputs = self.sess.run(None, feeds)
        last = outputs[0]
        if last.ndim == 3:
            pooled = self._mean_pool(last, batch["attention_mask"])
        else:
            pooled = last
        # L2 normalize
        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / np.clip(norm, 1e-9, None)
        if self.dim is None:
            self.dim = pooled.shape[-1]
        return pooled.tolist()
```

### 4.2 pipeline/reranker.py (ONNX 로드, 텍스트쌍 점수화)
```python
import os
from typing import List, Dict
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

class Reranker:
    def __init__(self):
        self.onnx_dir = os.getenv("RERANK_ONNX_DIR", os.getenv("RERANK_ONNX_PATH", "")).rstrip("/")
        self.model_name = os.getenv("RERANK_MODEL", "jinaai/jina-reranker-v2-base-multilingual")

        if self.onnx_dir and os.path.exists(os.path.join(self.onnx_dir, "model.int8.onnx")):
            self.sess = ort.InferenceSession(os.path.join(self.onnx_dir, "model.int8.onnx"), providers=["CPUExecutionProvider"])
            self.tok = AutoTokenizer.from_pretrained(self.onnx_dir, use_fast=True)
            self.fake = False
        else:
            # 간이 스코어(폴백)
            self.sess = None
            self.tok = None
            self.fake = True

    def rerank(self, query: str, candidates: List[str]) -> List[Dict]:
        if self.fake:
            # Overlap 기반 간이 스코어
            q_terms = set(query.lower().split())
            scored = []
            for i, t in enumerate(candidates):
                overlap = sum(1 for w in t.lower().split() if w in q_terms)
                score = min(0.99, 0.4 + 0.05*overlap)
                scored.append({"idx": i, "score": score})
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored

        # ONNX reranker: (query, doc) → logits
        batch = self.tok([query]*len(candidates), candidates, padding=True, truncation=True, max_length=512, return_tensors='np')
        feeds = {k: batch[k] for k in ["input_ids","attention_mask","token_type_ids"] if k in batch}
        outputs = self.sess.run(None, feeds)
        logits = outputs[0].reshape(-1)
        # 점수 정규화: 시그모이드
        scores = 1/(1+np.exp(-logits))
        order = np.argsort(-scores)
        return [{"idx": int(i), "score": float(scores[i])} for i in order]
```

> **환경변수 사용법**
> - `.env`에 다음을 지정:
> ```env
> EMBED_ONNX_DIR=/opt/models/emb
> RERANK_ONNX_DIR=/opt/models/rerank
> EMBED_MODEL=intfloat/multilingual-e5-small
> RERANK_MODEL=jinaai/jina-reranker-v2-base-multilingual
> ```

---

## 5) 컨테이너에서 모델 경로 마운트
`docker-compose.yml`의 **app**, **worker** 서비스에 다음 볼륨을 추가:
```yaml
    volumes:
      - ./pipeline:/workspace/pipeline:ro
      - /opt/models:/models:ro
```
그리고 `.env`에:
```env
EMBED_ONNX_DIR=/models/emb
RERANK_ONNX_DIR=/models/rerank
```

> 이미지를 빌드한 뒤에도 모델 교체만으로 재시작 가능 (`/opt/models` 갱신).

---

## 6) 절차 요약
1) (호스트) ONNX 내보내기: `sudo bash scripts/export_onnx.sh OUT_DIR=/opt/models`
2) (호스트) INT8 양자화: `python3 scripts/quantize_onnx.py --in /opt/models/.../model.onnx --out /opt/models/.../model.int8.onnx --per_channel`
3) `docker compose build app worker && docker compose up -d`
4) `/health` 확인 후 질의 테스트

---

## 7) 기대 효과(참고)
- 임베딩/재랭커 **CPU 사용량 30~60%** 절감, **지연 20~40%** 감소(모델/문장 길이에 따라 상이)
- 트래픽 급증 시에도 **조건부 재랭킹**과 결합하여 평균 지연 안정화

---

## 8) 문제 해결 팁
- 토큰 불일치 오류: 모델 디렉터리에 `tokenizer.json`/`vocab.txt`가 있는지 확인
- `token_type_ids` 미지원 모델: feeds에서 제거(코드가 자동 처리)
- 출력 텐서 이름이 다른 경우: `verify_onnx.py`로 실제 출력 구조 확인 후 embedder 풀링 로직 조정
- 양자화 후 정확도 하락 체감: `--per_channel` 유지, 필요 시 임베딩만 INT8, 재랭커는 FP16/FP32 유지








export_onnx.sh : HuggingFace 모델을 ONNX로 내보내기 (optimum-cli)

quantize_onnx.py : ONNX → INT8 동적 양자화

verify_onnx.py : 간단한 동작 확인

pipeline/embedder.py / pipeline/reranker.py : ONNX(INT8) 로딩·추론 코드로 패치

.env / docker-compose.yml에 필요한 경로·볼륨 가이드

바로 실행 순서

서버에서 sudo bash scripts/export_onnx.sh OUT_DIR=/opt/models

python3 scripts/quantize_onnx.py --in .../model.onnx --out .../model.int8.onnx --per_channel

.env에 EMBED_ONNX_DIR / RERANK_ONNX_DIR 설정 → docker compose build app worker && docker compose up -d