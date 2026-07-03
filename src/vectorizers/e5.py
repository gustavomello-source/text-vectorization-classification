"""
E5 Vectorization Strategy with Sliding Window
"""
import time
import numpy as np
import torch
from pandas import DataFrame
from transformers import AutoTokenizer, AutoModel
from .base_strategy import VectorizerStrategy
from src.utils.logger import log

class E5EmbeddingStrategy(VectorizerStrategy):
    """
    E5 embeddings com janela deslizante (sliding window) otimizada:
      - chunk_size = 512 (inclui special tokens)
      - stride = 510 (sem sobreposição)
      - agg = 'mean'
    Opera diretamente com token IDs (sem decodificação).
    """
    def __init__(self, model_name="intfloat/e5-base-v2", batch_size: int = 16):
        self.model_name = model_name
        self.batch_size = batch_size
        self.chunk_size = 512
        self.stride = 510
        self.agg = "mean"
        self.vectorization_time = 0.0  # Initialize timing attribute

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        log.info(f"E5EmbeddingStrategy initialized: model={self.model_name}, batch_size={self.batch_size}, "
                 f"chunk_size={self.chunk_size}, stride={self.stride}, device={self.device}")

    def tokenize_and_lemmatize(self, data: DataFrame, language: str = "english"):
        log.info("Skipping manual tokenization — handled by E5 tokenizer.")
        return data

    def _chunk_text_to_token_ids(self, text: str):
        """Divide texto em listas de token_ids (sem redecodificação)."""
        if text is None:
            return [[self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]]

        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        total_tokens = len(token_ids)
        usable = self.chunk_size - 2

        if usable <= 0:
            raise ValueError("chunk_size must be >= 3")
        if total_tokens == 0:
            return [[self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]]

        chunks = []
        i = 0
        while i < total_tokens:
            window = token_ids[i:i + usable]
            window = [self.tokenizer.cls_token_id] + window + [self.tokenizer.sep_token_id]
            chunks.append(window)
            if i + usable >= total_tokens:
                break
            i += self.stride
        return chunks

    def _chunk_texts(self, texts: list[str]):
        all_chunks, chunk_to_orig = [], []
        for idx, t in enumerate(texts):
            chunks = self._chunk_text_to_token_ids(t)
            all_chunks.extend(chunks)
            chunk_to_orig.extend([idx] * len(chunks))
        log.info(f"Built {len(all_chunks)} chunks from {len(texts)} documents (E5).")
        return all_chunks, chunk_to_orig

    def _encode_token_chunks(self, all_chunks: list[list[int]]) -> np.ndarray:
        """Codifica chunks (listas de token_ids) e aplica mean pooling."""
        if not all_chunks:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)

        all_embeddings = []
        n = len(all_chunks)
        for i in range(0, n, self.batch_size):
            batch_chunks = all_chunks[i:i + self.batch_size]
            max_len = max(len(x) for x in batch_chunks)

            padded = [x + [self.tokenizer.pad_token_id] * (max_len - len(x)) for x in batch_chunks]
            attention_mask = [[1] * len(x) + [0] * (max_len - len(x)) for x in batch_chunks]

            inputs = {
                "input_ids": torch.tensor(padded, dtype=torch.long, device=self.device),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=self.device)
            }

            with torch.no_grad():
                outputs = self.model(**inputs)

            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            batch_embs = summed / counts
            all_embeddings.append(batch_embs.cpu().numpy())

        result = np.vstack(all_embeddings)
        log.info(f"E5 encoded {n} chunks -> embeddings shape {result.shape}")
        return result

    def _encode_with_sliding_window(self, texts: list[str]) -> np.ndarray:
        all_chunks, chunk_to_orig = self._chunk_texts(texts)
        if not all_chunks:
            return np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)

        # Time only the actual encoding/vectorization step (not chunking)
        start_time = time.perf_counter()
        chunk_embeddings = self._encode_token_chunks(all_chunks)
        end_time = time.perf_counter()
        self.vectorization_time = end_time - start_time
        
        hidden_size = chunk_embeddings.shape[1]
        doc_embeddings = np.zeros((len(texts), hidden_size), dtype=np.float32)
        counts = np.zeros((len(texts),), dtype=np.int32)

        for idx_chunk, doc_idx in enumerate(chunk_to_orig):
            doc_embeddings[doc_idx] += chunk_embeddings[idx_chunk]
            counts[doc_idx] += 1

        nonzero = counts > 0
        doc_embeddings[nonzero] /= counts[nonzero][:, None]
        return doc_embeddings

    def vectorize(self, data: DataFrame, params: dict = None):
        if data.shape[1] != 1:
            raise ValueError("E5EmbeddingStrategy expects exactly one text column.")
        column = data.columns[0]
        texts = data[column].astype(str).tolist()
        log.info(f"E5 vectorization on column: {column} (sliding window mode)")
        X_vectorized = self._encode_with_sliding_window(texts)
        return X_vectorized, None, self.vectorization_time

    def transform(self, fitted_vectorizer, data: DataFrame):
        if data.shape[1] != 1:
            raise ValueError("E5EmbeddingStrategy expects exactly one text column.")
            
        column = data.columns[0]
        texts = data[column].astype(str).tolist()
        X_vectorized = self._encode_with_sliding_window(texts)
        return X_vectorized, self.vectorization_time