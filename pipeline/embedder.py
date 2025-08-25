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
