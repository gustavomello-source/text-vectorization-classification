"""
GPT-2 Vectorization Strategy with Sliding Window
"""
import time
import numpy as np
import torch
from pandas import DataFrame
from transformers import GPT2Tokenizer, GPT2Model
from .base_strategy import VectorizerStrategy
from src.utils.logger import log

class GPT2EmbeddingStrategy(VectorizerStrategy):
    """
    GPT-2 embeddings com janela deslizante (sliding window) otimizada:
      - chunk_size = 512 (inclui token <|endoftext|>)
      - stride = 511 (sem sobreposição)
      - agg = 'mean'
    GPT-2 usa apenas um token especial <|endoftext|> no final.
    """
    def __init__(self, model_name="gpt2", batch_size: int = 16):
        self.model_name = model_name
        self.batch_size = batch_size

        # parâmetros fixos
        self.chunk_size = 512
        self.stride = 511  # GPT-2 usa apenas 1 token especial
        self.agg = "mean"
        self.vectorization_time = 0.0  # Initialize timing attribute

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        
        # GPT-2 não tem pad_token por padrão, então adicionamos
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Importar GPT2Model da transformers
        self.model = GPT2Model.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        log.info(f"GPT2EmbeddingStrategy initialized: model={self.model_name}, batch_size={self.batch_size}, "
                 f"chunk_size={self.chunk_size}, stride={self.stride}, agg={self.agg}, device={self.device}")

    def tokenize_and_lemmatize(self, data, language: str = "english"):
        log.info("Skipping manual tokenization — handled by GPT-2 tokenizer.")
        return data

    def _chunk_text_to_token_ids(self, text: str):
        """
        Divide o texto em chunks de token_ids (sem redecodificação).
        Cada chunk inclui o token <|endoftext|> no final.
        """
        if text is None:
            return [[self.tokenizer.eos_token_id]]

        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        total_tokens = len(token_ids)
        usable = self.chunk_size - 1  # reserva 1 token para <|endoftext|>
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
        log.info(f"Built {len(all_chunks)} chunks from {len(texts)} documents (GPT-2).")
        return all_chunks, chunk_to_orig

    def _encode_token_chunks(self, all_chunks: list[list[int]]) -> np.ndarray:
        """
        Codifica chunks (listas de token_ids) usando GPT-2.
        """
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

            # média ponderada (mean pooling) usando a máscara
            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            batch_embeddings = summed / counts
            all_embeddings.append(batch_embeddings.cpu().numpy())

        result = np.vstack(all_embeddings)
        log.info(f"GPT-2 encoded {n} chunks -> embeddings shape {result.shape}")
        return result

    def _encode_with_sliding_window(self, texts: list[str]) -> np.ndarray:
        """Aplica janela deslizante e agrega embeddings de cada documento."""
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
        if nonzero.any():
            doc_embeddings[nonzero] /= counts[nonzero][:, None]

        return doc_embeddings

    def debug_print_chunk(self, text: str = None):
        """Print/log token ids/tokens/decoded for input text and first chunk (with and without special tokens)."""
        if text is None:
            text = "This is a short sample to show special tokens."

        try:
            # Full input: with special tokens
            enc_with = self.tokenizer.encode(text, add_special_tokens=True)
            tokens_with = self.tokenizer.convert_ids_to_tokens(enc_with)
            decoded_with = self.tokenizer.decode(enc_with)

            # Full input: raw (no special tokens)
            enc_raw = self.tokenizer.encode(text, add_special_tokens=False)
            tokens_raw = self.tokenizer.convert_ids_to_tokens(enc_raw)
            decoded_raw = self.tokenizer.decode(enc_raw)

            # Show chunking result
            chunks = self._chunk_text_to_token_ids(text)
            first_chunk_ids = chunks[0] if len(chunks) > 0 else []
            first_chunk_tokens = self.tokenizer.convert_ids_to_tokens(first_chunk_ids) if first_chunk_ids else []
            first_chunk_decoded = self.tokenizer.decode(first_chunk_ids) if first_chunk_ids else ""

            # Print to stdout for immediate visibility and log as well
            out = [
                ("INPUT (with special tokens) ids", enc_with),
                ("INPUT (with special tokens) tokens", tokens_with),
                ("INPUT (with special tokens) decoded", decoded_with),
                ("INPUT (raw) ids", enc_raw),
                ("INPUT (raw) tokens", tokens_raw),
                ("INPUT (raw) decoded", decoded_raw),
                ("FIRST CHUNK ids", first_chunk_ids),
                ("FIRST CHUNK tokens", first_chunk_tokens),
                ("FIRST CHUNK decoded", first_chunk_decoded),
                ("ALL CHUNKS count", len(chunks)),
            ]

            for title, value in out:
                print(f"{title}: {value}")
                log.info(f"GPT-2 debug - {title}: {value}")

            return {
                "input_with": {"ids": enc_with, "tokens": tokens_with, "decoded": decoded_with},
                "input_raw": {"ids": enc_raw, "tokens": tokens_raw, "decoded": decoded_raw},
                "first_chunk": {"ids": first_chunk_ids, "tokens": first_chunk_tokens, "decoded": first_chunk_decoded},
                "all_chunks": chunks
            }

        except Exception as e:
            log.error(f"GPT-2 debug_print_chunk error: {e}")
            print(f"GPT-2 debug_print_chunk error: {e}")
            return None

    def vectorize(self, data, params: dict = None):
        """Gera embeddings de documentos com sliding window."""
        if data.shape[1] != 1:
            raise ValueError("GPT2EmbeddingStrategy expects exactly one text column.")

        column = data.columns[0]
        texts = data[column].astype(str).tolist()
        log.info(f"GPT-2 vectorization on column: {column} (sliding window mode)")
        X_vectorized = self._encode_with_sliding_window(texts)
        return X_vectorized, None, self.vectorization_time

    def transform(self, fitted_vectorizer, data):
        if data.shape[1] != 1:
            raise ValueError("GPT2EmbeddingStrategy expects exactly one text column.")
            
        column = data.columns[0]
        texts = data[column].astype(str).tolist()
        X_vectorized = self._encode_with_sliding_window(texts)
        return X_vectorized, self.vectorization_time