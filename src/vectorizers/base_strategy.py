"""
Classe base abstrata para estratégias de vetorização
"""
from abc import ABC, abstractmethod
from pandas import DataFrame

class VectorizerStrategy(ABC):
    """Classe base abstrata para estratégias de vetorização"""
    
    @abstractmethod
    def vectorize(self, data: DataFrame, params: dict):
        """Vetoriza dados e retorna vetores, vetorizador ajustado e timing"""
        pass
    
    @abstractmethod
    def transform(self, fitted_vectorizer, data: DataFrame):
        """Transforma dados usando vetorizador ajustado e retorna vetores e timing"""
        pass

    @abstractmethod
    def tokenize_and_lemmatize(self, data: DataFrame, language: str = "english"):
        """Tokeniza e lemmatiza dados de texto"""
        pass