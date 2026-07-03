"""
Word2Vec Vectorization Strategy
"""

import time

import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pandas import DataFrame

from src.utils.logger import log

from .base_strategy import VectorizerStrategy


class Word2VecStrategy(VectorizerStrategy):
    def tokenize_and_lemmatize(self, data: DataFrame, language: str = "english"):
        if data.shape[1] != 1:
            log.warning(f"{self.__class__.__name__} expects exactly one text column.")
            raise ValueError(
                f"{self.__class__.__name__} expects exactly one text column."
            )

        column = data.columns[0]
        log.info(f"Tokenizing and lemmatizing column: {column}")

        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words(language))

        tokenized = []
        for text in data[column]:
            tokens = [
                lemmatizer.lemmatize(word.lower())
                for word in word_tokenize(text)
                if word.isalpha() and word.lower() not in stop_words
            ]
            tokenized.append(tokens)

        log.info("Tokenization and lemmatization completed.")
        return tokenized

    def vectorize(self, data: DataFrame, params: dict):
        if data.shape[1] != 1:
            log.warning("Word2VecStrategy expects exactly one text column.")
            raise ValueError("Word2VecStrategy expects exactly one text column.")

        column = data.columns[0]
        log.info(f"Word2Vec vectorization (training) on column: {column}")

        tokenized = self.tokenize_and_lemmatize(data)
        model = Word2Vec(vector_size=300, sentences=tokenized, seed=1)

        start_time = time.perf_counter()
        vectorized_data = np.array(
            [
                np.mean([model.wv[word] for word in tokens if word in model.wv], axis=0)
                if any(word in model.wv for word in tokens)
                else np.zeros(model.vector_size)
                for tokens in tokenized
            ]
        )
        end_time = time.perf_counter()

        return vectorized_data, model, end_time - start_time

    def transform(self, fitted_model: Word2Vec, data: DataFrame):
        if data.shape[1] != 1:
            log.warning("Word2VecStrategy expects exactly one text column.")
            raise ValueError("Word2VecStrategy expects exactly one text column.")

        column = data.columns[0]
        log.info(f"Word2Vec transformation on column: {column}")

        tokenized = self.tokenize_and_lemmatize(data)

        start_time = time.perf_counter()
        transformed_data = np.array(
            [
                np.mean(
                    [
                        fitted_model.wv[word]
                        for word in tokens
                        if word in fitted_model.wv
                    ],
                    axis=0,
                )
                if any(word in fitted_model.wv for word in tokens)
                else np.zeros(fitted_model.vector_size)
                for tokens in tokenized
            ]
        )
        end_time = time.perf_counter()

        return transformed_data, end_time - start_time
