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
