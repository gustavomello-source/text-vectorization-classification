"""
GloVe Vectorization Strategy
"""
import time
import numpy as np
from pandas import DataFrame
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from .base_strategy import VectorizerStrategy
from src.utils.logger import log

class GloVeStrategy(VectorizerStrategy):
    def __init__(self):
        self.glove_path = 'models/glove.6B.300d.txt'
        self.wv = None
        self.vector_size = None

    def tokenize_and_lemmatize(self, data: DataFrame, language: str = "english"):
        if data.shape[1] != 1:
            log.warning(f"{self.__class__.__name__} expects exactly one text column.")
            raise ValueError(f"{self.__class__.__name__} expects exactly one text column.")

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

    def _load_glove_vectors(self):
        log.info(f"Loading GloVe vectors from: {self.glove_path}")
        glove_model = {}
        with open(self.glove_path, "r", encoding="utf8") as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                vector = np.asarray(parts[1:], dtype="float32")
                glove_model[word] = vector
        self.vector_size = len(next(iter(glove_model.values())))
        self.wv = glove_model

    def vectorize(self, data, params):
        if data.shape[1] != 1:
            raise ValueError("GloVeStrategy expects exactly one text column.")
        if self.wv is None:
            self._load_glove_vectors()

        tokenized = self.tokenize_and_lemmatize(data)

        start_time = time.perf_counter()
        X = np.array([
            np.mean([self.wv[w] for w in tokens if w in self.wv] or
                    [np.zeros(self.vector_size)], axis=0)
            for tokens in tokenized
        ])
        end_time = time.perf_counter()
        return X, self.wv, end_time - start_time

    def transform(self, fitted_wv, data):
        if data.shape[1] != 1:
            raise ValueError("GloVeStrategy expects exactly one text column.")
        self.wv = fitted_wv
        self.vector_size = len(next(iter(fitted_wv.values())))

        tokenized = self.tokenize_and_lemmatize(data)
        start_time = time.perf_counter()
        X = np.array([
            np.mean([fitted_wv[w] for w in tokens if w in fitted_wv] or
                    [np.zeros(self.vector_size)], axis=0)
            for tokens in tokenized
        ])
        end_time = time.perf_counter()

        return X, end_time - start_time