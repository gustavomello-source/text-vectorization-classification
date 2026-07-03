"""
Estratégia de Vetorização TF-IDF
"""
import time
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from .base_strategy import VectorizerStrategy
from src.utils.logger import log

class TFIDFStrategy(VectorizerStrategy):
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

    def vectorize(self, data: DataFrame, params: dict):
        if data.shape[1] != 1:
            log.warning("TFIDFStrategy expects exactly one text column.")
            raise ValueError("TFIDFStrategy expects exactly one text column.")

        column = data.columns[0]
        log.info(f"TF-IDF vectorization on column: {column}")

        tokenized = self.tokenize_and_lemmatize(data)
        joined_texts = [" ".join(tokens) for tokens in tokenized]

        vectorizer = TfidfVectorizer()
        start_time = time.perf_counter()
        vectorized_data = vectorizer.fit_transform(joined_texts)
        end_time = time.perf_counter()

        return vectorized_data, vectorizer, end_time - start_time

    def transform(self, fitted_vectorizer, data: DataFrame):
        if data.shape[1] != 1:
            log.warning("TFIDFStrategy expects exactly one text column.")
            raise ValueError("TFIDFStrategy expects exactly one text column.")

        column = data.columns[0]
        log.info(f"TF-IDF transformation on column: {column}")

        tokenized = self.tokenize_and_lemmatize(data)
        joined_texts = [" ".join(tokens) for tokens in tokenized]

        start_time = time.perf_counter()
        transformed_data = fitted_vectorizer.transform(joined_texts)
        end_time = time.perf_counter()

        return transformed_data, end_time - start_time