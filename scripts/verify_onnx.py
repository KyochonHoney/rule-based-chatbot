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
