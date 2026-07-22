# Undergraduate thesis: Comparative Analysis of Text Vectorization Techniques for Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A comprehensive research project comparing traditional and transformer-based text vectorization techniques for text classification tasks. This project was developed as part of a graduate thesis to evaluate the effectiveness of various vectorization methods on different datasets.

## Overview

This project implements and compares **11 different text vectorization techniques**:

### Traditional Methods
- **Bag of Words (BoW)**: Simple word frequency representation
- **TF-IDF**: Term Frequency-Inverse Document Frequency weighting
- **Word2Vec**: Dense word embeddings using skip-gram/CBOW
- **GloVe**: Global Vectors for word representation
- **Doc2Vec**: Document-level embeddings

### Transformer-Based Methods
- **BERT**: Bidirectional Encoder Representations from Transformers
- **ALBERT**: A Lite BERT for self-supervised learning
- **RoBERTa**: Robustly optimized BERT pretraining
- **GPT-2**: Generative Pre-trained Transformer 2
- **E5**: Text Embeddings by Weakly-Supervised Contrastive Pre-training
- **Instructor**: Instruction-finetuned text embeddings

Each vectorization method is evaluated using multiple machine learning classifiers:
- Linear Support Vector Classifier (LinearSVC)
- Logistic Regression
- Random Forest

## Features

- **Modular Architecture**: Strategy pattern for easy addition of new vectorizers and models
- **Lazy Loading**: Memory-efficient loading of models only when needed
- **GPU Support**: Automatic CUDA detection and GPU memory management
- **Comprehensive Logging**: Detailed logging of all operations and performance metrics
- **Configurable**: INI-based configuration for datasets, preprocessing, and vectorizers
- **Results Export**: Automatic CSV export of all experimental results
- **Memory Management**: Automatic cleanup of GPU memory for transformer models

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for transformer models)
- At least 16GB RAM (32GB+ recommended for transformer models)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/gustavomello-source/TCC2.git
cd TCC2
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download required resources**

For Word2Vec and GloVe, you may need to download pre-trained vectors:
- Word2Vec: [Google News vectors](https://code.google.com/archive/p/word2vec/)
- GloVe: [Stanford GloVe](https://nlp.stanford.edu/projects/glove/)

## Supported Datasets

The project includes configuration for multiple text classification datasets, such as:

- **20 Newsgroups**: Multi-class news article classification
- **IMDB**: Binary sentiment analysis (movie reviews)
- **AG News**: News article categorization

Custom datasets can be added via the `config/dataset_config.ini` file.

## Usage

### Basic Usage

Run the main experiment:
```bash
python main.py
```

The script will:
1. Load configured datasets
2. Preprocess the text data
3. Apply each vectorization technique
4. Train and evaluate each classifier
5. Save results to `results/` directory

### Configuration

Edit configuration files in the `config/` directory:

- **`dataset_config.ini`**: Dataset paths and settings
- **`preprocess_config.ini`**: Text preprocessing options
- **`vectorizers_config.ini`**: Vectorizer-specific parameters

### Example: Custom Experiment

```python
from src.config import ConfigReader
from src.data import DatasetLoader
from src.preprocessing import Preprocessor
from src.vectorizers.vectorizer import TextVectorizer
from src.models import MLModel

# Load and preprocess data
dataset_loader = DatasetLoader("imdb")
data = dataset_loader.load()
preprocessor = Preprocessor()
clean_data = preprocessor.preprocess(data)

# Vectorize
vectorizer = TextVectorizer("BERT")
X_train, _, _ = vectorizer.vectorize_data(clean_data)

# Train model
model = MLModel("LogisticRegression")
model.train(X_train, y_train)
```

## Project Structure

```
TCC2/
├── src/                          # Source code
│   ├── config/                   # Configuration management
│   │   ├── config_reader.py      # INI file reader
│   │   └── __init__.py
│   ├── data/                     # Data loading
│   │   ├── dataset_loader.py     # Dataset loader
│   │   └── __init__.py
│   ├── models/                   # ML models
│   │   ├── base_strategy.py      # Base model strategy
│   │   ├── linear_svc.py         # Linear SVC implementation
│   │   ├── logistic_regression.py # Logistic Regression
│   │   ├── random_forest.py      # Random Forest
│   │   ├── _ml_model.py          # Main model class
│   │   ├── models.py             # Backward compatibility
│   │   └── __init__.py
│   ├── preprocessing/            # Text preprocessing
│   │   ├── preprocess.py         # Preprocessing pipeline
│   │   └── __init__.py
│   ├── utils/                    # Utilities
│   │   ├── logger.py             # Logging configuration
│   │   ├── results.py            # Results processing
│   │   └── __init__.py
│   ├── vectorizers/              # Vectorization strategies
│   │   ├── base_strategy.py      # Base vectorizer strategy
│   │   ├── bow.py                # Bag of Words
│   │   ├── tfidf.py              # TF-IDF
│   │   ├── word2vec.py           # Word2Vec
│   │   ├── glove.py              # GloVe
│   │   ├── doc2vec.py            # Doc2Vec
│   │   ├── bert.py               # BERT
│   │   ├── albert.py             # ALBERT
│   │   ├── roberta.py            # RoBERTa
│   │   ├── gpt2.py               # GPT-2
│   │   ├── e5.py                 # E5 Embeddings
│   │   ├── instructor.py         # Instructor Embeddings
│   │   ├── vectorizer.py         # Main vectorizer class
│   │   └── __init__.py
│   └── __init__.py
├── config/                       # Configuration files
│   ├── dataset_config.ini
│   ├── preprocess_config.ini
│   └── vectorizers_config.ini
├── results/                      # Experiment results (auto-generated)
├── logs/                         # Log files (auto-generated)
├── tests/                        # Unit tests (future)
├── notebooks/                    # Jupyter notebooks (future)
├── docs/                         # Documentation (future)
├── main.py                       # Main execution script
├── setup.py                      # Package setup
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Project metadata
├── .gitignore                    # Git ignore rules
├── .gitattributes                # Git attributes
├── LICENSE                       # MIT License
└── README.md                     # This file
```

## Methodology

### Preprocessing Pipeline
1. Text normalization (lowercase, remove special characters)
2. Tokenization
3. Stop word removal (configurable)
4. Stemming/Lemmatization (optional)

### Evaluation Metrics
- Accuracy
- Precision, Recall, F1-Score (macro and weighted)
- Training and inference time
- Memory usage

### Cross-Validation
- Stratified train-test split
- Multiple random seeds for reproducibility

## Results

Results are automatically saved in CSV format in the `results/` directory with the following information:

- Dataset name
- Vectorization method
- Model type
- All performance metrics
- Execution times
- Timestamp

### Adding a New Vectorizer

1. Create a new file in `src/vectorizers/` (e.g., `new_vectorizer.py`)
2. Inherit from `VectorizerStrategy`
3. Implement `vectorize()` and `transform()` methods
4. Register in `TextVectorizer._strategy_classes`
5. Add configuration in `config/vectorizers_config.ini`

### Adding a New Model

1. Create a new file in `src/models/` (e.g., `new_model.py`)
2. Inherit from `ModelStrategy`
3. Implement `train()` and `predict()` methods
4. Register in `MLModel._model_classes`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Pre-trained models from Hugging Face Transformers
- Word embeddings from Word2Vec and GloVe projects
- Dataset providers
- Open-source community

## Contact

**Gustavo Mello**
- GitHub: [@gustavomello-source](https://github.com/gustavomello-source)
- Repository: [text-vectorization-classification](https://github.com/gustavomello-source/text-vectorization-classification)

---
