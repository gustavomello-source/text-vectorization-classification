"""
Doc2Vec Vectorization Strategy
"""
import time
import numpy as np
from typing import Any
from pandas import DataFrame
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from .base_strategy import VectorizerStrategy
from src.utils.logger import log

class Doc2VecStrategy(VectorizerStrategy):
    def __init__(self, vector_size: int = 300):
        self.vector_size = vector_size
        self.model = None

    def tokenize_and_lemmatize(self, data: DataFrame, language: str = "english"):
        """Perform tokenization + lemmatization if needed."""
        log.info("Tokenizing and lemmatizing for Doc2Vec.")
        # Example: assume each cell is a string of text
        return data[data.columns[0]].apply(lambda text: text.split())  # Replace with your actual preprocessing

    def vectorize(self, data: DataFrame, params: dict = None):
        if data.shape[1] != 1:
            raise ValueError("Doc2VecStrategy expects exactly one text column.")

        tokenized = self.tokenize_and_lemmatize(data)
        tagged_docs = [TaggedDocument(words=tokens, tags=[i]) for i, tokens in enumerate(tokenized)]

        log.info("Training Doc2Vec model...")
        self.model = Doc2Vec(
            documents=tagged_docs,
            vector_size=self.vector_size,
            seed=1
        )
        start_time = time.perf_counter()
        X = np.array([self.model.dv[i] for i in range(len(tagged_docs))])
        end_time = time.perf_counter()
        return X, self.model, end_time - start_time

    def transform(self, fitted_model: Any, data: DataFrame):
        tokenized = self.tokenize_and_lemmatize(data)
        log.info("Inferring vectors for new documents with Doc2Vec...")
        start_time = time.perf_counter()
        X = np.array([fitted_model.infer_vector(tokens) for tokens in tokenized])
        end_time = time.perf_counter()
        return X, end_time - start_time