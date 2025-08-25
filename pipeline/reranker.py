import os
from typing import List, Dict
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

class Reranker:
    def __init__(self):
        self.onnx_dir = os.getenv("RERANK_ONNX_DIR", os.getenv("RERANK_ONNX_PATH", "")).rstrip("/")
        self.model_name = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

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
