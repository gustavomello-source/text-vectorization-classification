"""
TCC2 - Comparative analysis of vectorization text techniques applied in classification

A research project comparing different text vectorization techniques (BoW, TF-IDF,
Word2Vec, Doc2Vec, GloVe, BERT, RoBERTa, GPT-2, ALBERT, E5, Instructor) for
text classification tasks using various machine learning models.
"""

__version__ = "1.0.0"
__author__ = "Gustavo Mello"

# Lazy imports - components are imported only when accessed
__all__ = [
    "ConfigReader",
    "DatasetLoader",
    "Preprocessor",
    "MLModel",
    "TextVectorizer",
]


def __getattr__(name):
    """Lazy loading of modules to avoid loading all dependencies at import time."""
    if name == "ConfigReader":
        from src.config.config_reader import ConfigReader

        return ConfigReader
    elif name == "DatasetLoader":
        from src.data.dataset_loader import DatasetLoader

        return DatasetLoader
    elif name == "Preprocessor":
        from src.preprocessing.preprocess import Preprocessor

        return Preprocessor
    elif name == "MLModel":
        from src.models._ml_model import MLModel

        return MLModel
    elif name == "TextVectorizer":
        from src.vectorizers.vectorizer import TextVectorizer

        return TextVectorizer
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
