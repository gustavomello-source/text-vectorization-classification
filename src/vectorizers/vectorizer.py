"""
Main TextVectorizer class that uses modular vectorization strategies.

Supports multiple vectorization methods including traditional (BoW, TF-IDF, Word2Vec, GloVe, Doc2Vec)
and transformer-based approaches (BERT, ALBERT, RoBERTa, GPT-2, E5, InstructorEmbedding).

Uses the Strategy pattern for extensibility and lazy-loading for memory efficiency.
Includes automatic GPU memory cleanup for transformer models.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from pandas import DataFrame

from src.utils.logger import log

from .albert import ALBERTStrategy
from .bert import BERTStrategy
from .bow import BoWStrategy
from .doc2vec import Doc2VecStrategy
from .e5 import E5EmbeddingStrategy
from .glove import GloVeStrategy
from .gpt2 import GPT2EmbeddingStrategy
from .instructor import InstructorEmbeddingStrategy
from .roberta import RoBERTaStrategy

# Importa todas as estratégias de vetorização
from .tfidf import TFIDFStrategy
from .word2vec import Word2VecStrategy


class TextVectorizer:
    def __init__(self, method: str, vectorizer_params: dict = None):
        self._strategy_classes = {
            "BoW": BoWStrategy,
            "TF-IDF": TFIDFStrategy,
            "Word2Vec": Word2VecStrategy,
            "GloVe": GloVeStrategy,
            "Doc2Vec": Doc2VecStrategy,
            "BERT": BERTStrategy,
            "ALBERT": ALBERTStrategy,
            "RoBERTa": RoBERTaStrategy,
            "GPT2": GPT2EmbeddingStrategy,
            "E5": E5EmbeddingStrategy,
            "InstructorEmbedding": InstructorEmbeddingStrategy,
        }

        self._vectorizer_params = (
            vectorizer_params if vectorizer_params is not None else {}
        )

        if method not in self._strategy_classes:
            available_methods = list(self._strategy_classes.keys())
            raise ValueError(
                f"Unsupported method '{method}'. Available methods: {available_methods}"
            )

        self._method = method
        self._strategy = None

        log.info(f"Initializing vectorizer: {method}")

    def cleanup(self):
        """Limpa estratégia carregada e libera memória"""
        if self._strategy is not None:
            log.info(f"Cleaning up {self._method} strategy...")

            if hasattr(self._strategy, "model") and self._strategy.model is not None:
                if hasattr(self._strategy.model, "cpu"):
                    self._strategy.model.cpu()  # Move para CPU
                del self._strategy.model

            if hasattr(self._strategy, "tokenizer"):
                del self._strategy.tokenizer

            del self._strategy
            self._strategy = None

            # Limpa cache da GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            import gc

            gc.collect()  # Força coleta de lixo

            log.info(f"{self._method} strategy cleaned up")

    def __del__(self):
        """Destrutor para garantir limpeza"""
        self.cleanup()

    def _get_strategy(self):
        """Inicialização dos vetorizadores"""
        if self._strategy is None:
            log.info(f"Loading {self._method} strategy...")
            strategy_class = self._strategy_classes[self._method]

            if self._method in [
                "BERT",
                "ALBERT",
                "RoBERTa",
                "E5",
                "InstructorEmbedding",
            ]:
                self._strategy = strategy_class()
            elif self._method == "Doc2Vec":
                self._strategy = strategy_class()
            else:
                self._strategy = strategy_class()

        return self._strategy

    def vectorize_data(self, data: DataFrame) -> tuple[DataFrame, any, float]:
        log.info(f"Vectorizing data using method: {self.method}")

        try:
            strategy = self._get_strategy()  # Only load the needed strategy
            vectors, vectorizer, vectorize_time = strategy.vectorize(
                data, self.vectorizer_params
            )

            if vectors is not None:
                if isinstance(vectors, list) and len(vectors) > 0:
                    if len(vectors) == 1:
                        combined_matrix = vectors[0]
                    else:
                        combined_matrix = sp.hstack(vectors)

                    vectors_df = pd.DataFrame(combined_matrix.toarray())
                    log.info(f"Vectorization complete. Shape: {vectors_df.shape}")
                    return vectors_df, vectorizer, vectorize_time
                else:
                    log.info("Vectorization complete.")
                    return vectors, vectorizer, vectorize_time
            else:
                log.warning(f"Vectorization returned None for method: {self.method}")
                return None, None, None
        except Exception as e:
            log.error(f"Error during {self.method} vectorization: {e}")
            return None, None, None

    def transform_data(
        self, fitted_vectorizer, data: DataFrame
    ) -> tuple[DataFrame, float]:
        """Transform new data using a fitted vectorizer"""
        log.info(f"Transforming data using fitted {self.method} vectorizer")

        if fitted_vectorizer is None:
            try:
                log.warning(
                    "Fitted vectorizer is None, trying to vectorize remaining data"
                )
                vectors_df, _, vectorize_time = self.vectorize_data(data)
                return pd.DataFrame(self.vectorize_data(data)[0]), vectorize_time
            except Exception as e:
                log.error(f"Error during {self.method} transformation: {e}")
                return None, None
        try:
            strategy = self._get_strategy()
            transformed_data, transform_time = strategy.transform(
                fitted_vectorizer, data
            )

            if hasattr(transformed_data, "toarray"):
                vectors_df = pd.DataFrame(transformed_data.toarray())
            elif isinstance(transformed_data, np.ndarray):
                vectors_df = pd.DataFrame(transformed_data)
            elif isinstance(transformed_data, list):
                if len(transformed_data) == 1:
                    combined_matrix = transformed_data[0]
                else:
                    combined_matrix = sp.hstack(transformed_data)
                vectors_df = pd.DataFrame(combined_matrix.toarray())
            else:
                vectors_df = transformed_data

            log.info(f"Transform complete. Shape: {vectors_df.shape}")
            return vectors_df, transform_time

        except Exception as e:
            log.error(f"Error during {self.method} transformation: {e}")
            return None, None

    @property
    def method(self) -> str:
        return self._method

    @method.setter
    def method(self, value) -> None:
        if not isinstance(value, str):
            raise ValueError("Method must be a string")
        if value not in self._strategy_classes:
            available = list(self._strategy_classes.keys())
            raise ValueError(f"Unsupported method '{value}'. Available: {available}")
        self._method = value

    @property
    def vectorizer_params(self) -> dict:
        return self._vectorizer_params

    @vectorizer_params.setter
    def vectorizer_params(self, value) -> None:
        if not isinstance(value, dict):
            raise ValueError("Vectorizer parameters must be a dictionary")
        self._vectorizer_params = value
