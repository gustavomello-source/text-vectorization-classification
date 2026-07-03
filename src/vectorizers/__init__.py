"""
Vectorization strategies package
"""

from .base_strategy import VectorizerStrategy
from .tfidf import TFIDFStrategy
from .bow import BoWStrategy
from .word2vec import Word2VecStrategy
from .glove import GloVeStrategy
from .doc2vec import Doc2VecStrategy
from .bert import BERTStrategy
from .albert import ALBERTStrategy
from .roberta import RoBERTaStrategy
from .gpt2 import GPT2EmbeddingStrategy
from .e5 import E5EmbeddingStrategy
from .instructor import InstructorEmbeddingStrategy
from .vectorizer import TextVectorizer

__all__ = [
    'VectorizerStrategy',
    'TFIDFStrategy',
    'BoWStrategy', 
    'Word2VecStrategy',
    'GloVeStrategy',
    'Doc2VecStrategy',
    'BERTStrategy',
    'ALBERTStrategy',
    'RoBERTaStrategy',
    'GPT2EmbeddingStrategy',
    'E5EmbeddingStrategy',
    'InstructorEmbeddingStrategy',
    'TextVectorizer'
]

# __all__ = [
#     'ModelStrategy',
#     'LinearSVCStrategy',
#     'LogisticRegressionStrategy',
#     'RandomForestStrategy',
#     'MLModel',
#     'TextCNNStrategy'
# ]