"""
Instructor Embedding Vectorization Strategy with Sliding Window
"""
import time
import numpy as np
import torch
from pandas import DataFrame
from sentence_transformers import SentenceTransformer
from .base_strategy import VectorizerStrategy
from src.utils.logger import log

class InstructorEmbeddingStrategy(VectorizerStrategy):
    """
    Instructor embeddings com janela deslizante (sliding window) otimizada:
      - chunk_size = 512 (inclui token </s>)
      - stride = 511 (sem sobreposição)
      - agg = 'mean'
    Evita recompressão desnecessária ao trabalhar direto com token IDs.
    """
    def __init__(self, model_name: str = "hkunlp/instructor-xl", batch_size: int = 16):
        self.model_name = model_name
        self.batch_size = batch_size

        # parâmetros fixos
        self.chunk_size = 512
        self.stride = 511  # T5 usa apenas </s> como token especial
        self.agg = "mean"
        self.vectorization_time = 0.0  # Initialize timing attribute

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(self.model_name, device=str(self.device))
        self.model.eval()

        # acesso direto ao tokenizador
        self.tokenizer = self.model.tokenizer

        log.info(f"InstructorEmbeddingStrategy initialized: model={self.model_name}, batch_size={self.batch_size}, "
                 f"chunk_size={self.chunk_size}, stride={self.stride}, agg={self.agg}, device={self.device}")

    def tokenize_and_lemmatize(self, data, language: str = "english"):
        log.info("Skipping manual tokenization — handled by Instructor tokenizer.")
        return data

    def _chunk_text_to_token_ids(self, text: str):
        """
        Divide o texto em chunks de token_ids (sem redecodificação).
        Cada chunk inclui o token </s> no final.
        """
        if text is None:
            return [[self.tokenizer.eos_token_id]]

        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        total_tokens = len(token_ids)
        usable = self.chunk_size - 1  # reserva 1 token para </s>
        if usable <= 0:
            raise ValueError("chunk_size must be >= 2")

        if total_tokens == 0:
            return [[self.tokenizer.eos_token_id]]

        chunks = []
        i = 0
        while i < total_tokens:
            window = token_ids[i:i + usable]
            window = window + [self.tokenizer.eos_token_id]
            chunks.append(window)
            if i + usable >= total_tokens:
                break
            i += self.stride
        return chunks

    def _chunk_texts(self, texts: list[str]):
        """Retorna lista de chunks (token_ids) e mapeamento chunk->documento."""
        all_chunks, chunk_to_orig = [], []
        for idx, t in enumerate(texts):
            chunks = self._chunk_text_to_token_ids(t)
            all_chunks.extend(chunks)
            chunk_to_orig.extend([idx] * len(chunks))
        log.info(f"Built {len(all_chunks)} chunks from {len(texts)} documents (Instructor).")
        return all_chunks, chunk_to_orig

    def _decode_chunks_to_texts(self, all_chunks: list[list[int]]) -> list[str]:
        """
        Converte listas de token_ids de volta para strings únicas.
        Necessário porque SentenceTransformer espera texto, não token IDs.
        """
        return [self.tokenizer.decode(chunk, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for chunk in all_chunks]

    def _encode_token_chunks(self, all_chunks: list[list[int]]) -> np.ndarray:
        """
        Codifica chunks usando o modelo Instructor (SentenceTransformer).
        """
        if not all_chunks:
            return np.zeros((0, 512), dtype=np.float32)  # dimensão padrão do Instructor-large

        # converte os token_ids para texto apenas uma vez
        chunk_texts = self._decode_chunks_to_texts(all_chunks)
        n = len(chunk_texts)
        all_embeddings = []

        log.info(f"Instructor: encoding {n} chunks in batches of {self.batch_size}...")
        for i in range(0, n, self.batch_size):
            batch_texts = chunk_texts[i:i + self.batch_size]
            embeddings = self.model.encode(
                batch_texts,
                prompt="Represent this text for classification: ",
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_tensor=False,
                normalize_embeddings=False
            )
            all_embeddings.append(np.array(embeddings, dtype=np.float32))

        result = np.vstack(all_embeddings)
        log.info(f"Instructor encoded {n} chunks -> embeddings shape {result.shape}")
        return result

    def _encode_with_sliding_window(self, texts: list[str]) -> np.ndarray:
        """Aplica janela deslizante e agrega embeddings de cada documento."""
        all_chunks, chunk_to_orig = self._chunk_texts(texts)
        if not all_chunks:
            return np.zeros((len(texts), 512), dtype=np.float32)

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
        if nonzero.any():
            doc_embeddings[nonzero] /= counts[nonzero][:, None]

        return doc_embeddings

    def vectorize(self, data, params: dict = None):
        """Gera embeddings de documentos com sliding window."""
        if data.shape[1] != 1:
            raise ValueError("InstructorEmbeddingStrategy expects exactly one text column.")

        column = data.columns[0]
        texts = data[column].astype(str).tolist()
        log.info(f"Instructor vectorization on column: {column} (sliding window mode)")
        X_vectorized = self._encode_with_sliding_window(texts)
        return X_vectorized, None, self.vectorization_time

    def transform(self, fitted_vectorizer, data):
        if data.shape[1] != 1:
            raise ValueError("InstructorEmbeddingStrategy expects exactly one text column.")
            
        column = data.columns[0]
        texts = data[column].astype(str).tolist()
        X_vectorized = self._encode_with_sliding_window(texts)
        return X_vectorized, self.vectorization_time