import re

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

from src.utils.logger import log


class Preprocessor:
    """Applies a configurable pipeline of text preprocessing stages to train/test data.

    The stages to run are read from a ``preprocess_config`` dict containing a
    single ``stages`` key with a comma-separated list of stage names. Target
    labels are encoded and normalized independently from the text features.
    """

    def __init__(
        self,
        X_train: DataFrame = None,
        X_test: DataFrame = None,
        y_train: DataFrame = None,
        y_test: DataFrame = None,
        preprocess_config: dict = None,
        encoding: str = None,
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.preprocess_stages = self.set_processing_stages(preprocess_config)
        self.encoding = encoding

    def set_processing_stages(self, preprocess_config: dict) -> list[str]:
        """Parse the comma-separated ``stages`` string into a list of stage names."""
        stages_str = preprocess_config.get("stages", None)

        selected_stages = [
            stage.strip() for stage in stages_str.split(",") if stage.strip()
        ]

        log.info(f"-------------- Preprocessing stages: {selected_stages}")

        return selected_stages

    def preprocess_data(self) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        """Preprocess both train and test data"""
        log.info(
            f"-------------- Starting preprocessing with {len(self.preprocess_stages)} stages"
        )

        _X_train_processed = None
        _X_test_processed = None

        _y_train, _y_test = self._apply_target_preprocessing()
        log.info("-------------- Processing training data...")
        _X_train_processed = self._apply_preprocessing_stages(self.X_train.copy())

        log.info("-------------- Processing test data...")
        _X_test_processed = self._apply_preprocessing_stages(self.X_test.copy())

        log.info("-------------- Preprocessing completed successfully")
        return _X_train_processed, _y_train, _X_test_processed, _y_test

    def _apply_preprocessing_stages(self, df: DataFrame) -> DataFrame:
        """Apply all preprocessing stages to a DataFrame"""
        for i, stage in enumerate(self.preprocess_stages, 1):
            log.info(f"Applying stage {i}/{len(self.preprocess_stages)}: {stage}")

            try:
                if stage == "lowercase":
                    df = self._apply_lowercase(df)
                elif stage == "remove_punctuation":
                    df = self._apply_remove_punctuation(df)
                elif stage == "light_remove_punctuation":
                    df = self._apply_light_remove_punctuation(df)
                elif stage == "remove_stopwords_eng":
                    df = self._apply_remove_stopwords_eng(df)
                elif stage == "lemmatization_eng":
                    df = self._apply_lemmatization_eng(df)
                elif stage == "remove_special_characters":
                    df = self._apply_remove_special_characters(df)
                elif stage == "remove_headers":
                    df = self._apply_remove_headers(df)
                elif stage == "remove_urls":
                    df = self._apply_remove_urls(df)
                elif stage == "remove_handles":
                    df = self._apply_remove_handles(df)
                elif stage == "remove_reply_markers":
                    df = self._apply_remove_reply_markers(df)
                elif stage == "normalize_whitespace":
                    df = self._apply_normalize_whitespace(df)
                elif stage == "normalize_numbers":
                    df = self._apply_normalize_numbers(df)
                elif stage == "remove_emojis":
                    df = self._apply_remove_emojis(df)
                elif stage == "remove_hashtags":
                    df = self._apply_remove_hashtags(df)
                elif stage == "remove_duplicates":
                    df = self._apply_remove_duplicates(df)
                elif stage == "tokenization":
                    df = self._apply_tokenization(df)
                else:
                    log.warning(f"Unknown preprocessing stage: {stage}")
                    continue

                log.info(f"Completed preprocess stage: {stage}")

            except Exception as e:
                log.error(f"Error in preprocessing stage '{stage}': {e}")
                continue

        return DataFrame(df)

    def _apply_target_preprocessing(self) -> tuple[DataFrame, DataFrame]:
        """Apply preprocessing to target variables if needed"""

        y_train_encoded = self._encode_if_categorical(self.y_train)
        y_test_encoded = self._encode_if_categorical(self.y_test)

        _y_train_processed, _y_test_processed = self._apply_normalize_target(
            y_train_encoded, y_test_encoded
        )

        return _y_train_processed, _y_test_processed

    def _apply_remove_duplicates(self, df):
        """Remove duplicate rows from the DataFrame"""
        initial_count = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        final_count = len(df)
        log.info(f"Removed {initial_count - final_count} duplicate rows")
        return df

    def _apply_normalize_numbers(self, df):
        """Replace numbers with a <num> token"""

        def normalize_numbers(text):
            # Replace any integer/decimal number with <num>
            text = re.sub(r"\b\d+(\.\d+)?\b", "<num>", text)
            # Also handle cases like "1,000" or "3.14"
            text = re.sub(r"\b\d{1,3}(,\d{3})*(\.\d+)?\b", "<num>", text)
            return text

        return df.map(
            lambda text: normalize_numbers(text) if isinstance(text, str) else text
        )

    def _apply_remove_emojis(self, df):
        """Remove emojis from text"""
        emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"  # emoticons
            "\U0001f300-\U0001f5ff"  # symbols & pictographs
            "\U0001f680-\U0001f6ff"  # transport & map symbols
            "\U0001f1e0-\U0001f1ff"  # flags (iOS)
            "\U00002702-\U000027b0"
            "\U000024c2-\U0001f251"
            "]+",
            flags=re.UNICODE,
        )

        return df.map(
            lambda text: emoji_pattern.sub(r"", text) if isinstance(text, str) else text
        )

    def _apply_remove_hashtags(self, df):
        """Remove hashtags from text"""
        return df.map(
            lambda text: re.sub(r"#\w+", "", text) if isinstance(text, str) else text
        )

    def _apply_light_remove_punctuation(self, df):
        """Remove punctuation but keep sentence delimiters (.?!)."""
        return df.map(
            lambda text: (
                re.sub(r"\s+", " ", re.sub(r"[^\w\s\.\?\!]", "", text)).strip()
                if isinstance(text, str)
                else text
            )
        )

    def _apply_normalize_whitespace(self, df):
        """Normalize whitespace without collapsing sentence structure."""
        return df.map(
            lambda x: re.sub(r"\s+", " ", x).strip() if isinstance(x, str) else x
        )

    def _apply_lowercase(self, df):
        """Apply lowercase transformation to string columns"""
        return df.map(lambda x: x.lower() if isinstance(x, str) else x)

    def _apply_remove_punctuation(self, df):
        """Remove punctuation from string columns while handling newlines properly"""

        def clean_punctuation(text):
            text = re.sub(r"\n[a-zA-Z]{1,3}[>:\s]*", " ", text)
            text = re.sub(r"\n+", " ", text)
            text = re.sub(r"[^\w\s]", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        return df.map(
            lambda text: clean_punctuation(text) if isinstance(text, str) else text
        )

    def _apply_remove_stopwords_eng(self, df):
        stop_words = set(stopwords.words("english"))
        return df.map(
            lambda text: (
                " ".join(
                    [word for word in text.split() if word.lower() not in stop_words]
                )
                if isinstance(text, str)
                else text
            )
        )

    def _apply_lemmatization_eng(self, df):
        lemmatizer = WordNetLemmatizer()
        return df.map(
            lambda text: (
                " ".join([lemmatizer.lemmatize(word) for word in text.split()])
                if isinstance(text, str)
                else text
            )
        )

    def _apply_remove_special_characters(self, df):
        """Remove special characters from string columns"""
        return df.map(
            lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", x) if isinstance(x, str) else x
        )

    def _apply_remove_headers(self, df):
        """Remove email headers from newsgroup posts"""

        def clean_headers(text):
            if not isinstance(text, str):
                return text

            # Split text into lines
            lines = text.split("\n")
            content_lines = []
            header_section = True

            for line in lines:
                # Check if we're still in header section
                if header_section:
                    # Headers typically start with "Field-Name:" format
                    if re.match(r"^[A-Za-z-]+:", line.strip()):
                        continue
                    # Empty line often marks end of headers
                    elif line.strip() == "":
                        header_section = False
                        continue
                    # If line doesn't look like header, we're in content
                    else:
                        header_section = False

                # Add content lines (skip empty lines at start of content)
                if not header_section and line.strip():
                    content_lines.append(line)

            # Join with spaces instead of newlines to avoid the '\nra' issue
            content = " ".join(content_lines)

            # Remove common newsgroup artifacts
            content = re.sub(r"\bra>\s*", "", content)  # Remove 'ra>' patterns
            content = re.sub(
                r"\b[a-zA-Z]{1,3}>\s*", "", content
            )  # Remove other short reply markers
            content = re.sub(r"\s+", " ", content).strip()  # Clean up spacing

            return content

        return df.map(lambda x: clean_headers(x) if isinstance(x, str) else x)

    def _apply_remove_urls(self, df):
        """Remove URLs from text"""

        def clean_urls(text):
            # Remove HTTP/HTTPS URLs
            text = re.sub(r'https?://[^\s<>"{}|\\^`\[\]]*', "", text)
            # Remove FTP URLs
            text = re.sub(r'ftp://[^\s<>"{}|\\^`\[\]]*', "", text)
            # Remove email addresses (optional - might want to keep for some datasets)
            text = re.sub(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text
            )
            # Remove www. URLs without protocol
            text = re.sub(r'www\.[^\s<>"{}|\\^`\[\]]*', "", text)
            # Clean up extra whitespace
            text = re.sub(r"\s+", " ", text).strip()
            return text

        return df.map(lambda text: clean_urls(text) if isinstance(text, str) else text)

    def _apply_remove_handles(self, df):
        """Remove social media handles and mentions"""

        def clean_handles(text):
            # Remove @mentions
            text = re.sub(r"@\w+", "", text)
            # Remove hashtags (optional - you might want to keep the text part)
            text = re.sub(r"#\w+", "", text)
            # Clean up extra whitespace
            text = re.sub(r"\s+", " ", text).strip()
            return text

        return df.map(
            lambda text: clean_handles(text) if isinstance(text, str) else text
        )

    def _apply_remove_reply_markers(self, df):
        """Remove newsgroup reply markers and quoted text artifacts"""

        def clean_reply_markers(text):
            if not isinstance(text, str):
                return text

            # Remove common reply patterns like 'ra>', 'kb>', etc.
            text = re.sub(r"\b[a-zA-Z]{1,3}>\s*", "", text)

            # Remove quoted text markers like '|>', '>', etc.
            text = re.sub(r"^[>|]\s*", "", text, flags=re.MULTILINE)

            # Remove lines that are just reply markers
            lines = text.split("\n")
            clean_lines = [
                line for line in lines if not re.match(r"^\s*[a-zA-Z]{1,3}>\s*$", line)
            ]
            text = "\n".join(clean_lines)

            # Clean up extra spaces
            text = re.sub(r"\s+", " ", text).strip()

            return text

        return df.map(lambda x: clean_reply_markers(x) if isinstance(x, str) else x)

    def _apply_tokenization(self, df):
        """Tokenize text into words using NLTK"""

        def tokenize_text(text):
            if isinstance(text, str):
                tokens = word_tokenize(text)
                return " ".join(tokens)
            return text

        return df.map(tokenize_text)

    def _encode_if_categorical(self, y):
        if y is not None:
            encoder = LabelEncoder()
            if isinstance(y, pd.DataFrame):
                y_encoded = y.apply(
                    lambda col: (
                        encoder.fit_transform(col) if col.dtype == "object" else col
                    )
                )
            else:
                y_encoded = encoder.fit_transform(y)
            log.info(f"Encoded labels with classes: {list(encoder.classes_)}")
            return pd.DataFrame(y_encoded, columns=["target"])
        return y

    def _apply_normalize_target(
        self, y_train: DataFrame, y_test: DataFrame
    ) -> tuple[DataFrame, DataFrame]:
        """
        Normalizes the target variables to have consistent labels.
        Decreases the value of all target labels by min value, so that the lowest label starts from 0.

        Returns:
            Tuple of normalized y_train and y_test as DataFrames
        """
        _y_train_normalized, _y_test_normalized = None, None

        if y_train is not None:
            try:
                min_label = (
                    y_train.min().values[0]
                    if isinstance(y_train, DataFrame)
                    else y_train.min()
                )
                _y_train_normalized = y_train - min_label
                log.info(f"Normalized y_train using min label {min_label}")
            except Exception as e:
                log.error(f"Error normalizing y_train: {e}")
                _y_train_normalized = y_train

        if y_test is not None:
            try:
                _y_test_normalized = y_test - min_label  # Use the same min_label
                log.info("Normalized y_test using training min label")
            except Exception as e:
                log.error(f"Error normalizing y_test: {e}")
                _y_test_normalized = y_test

        return DataFrame(_y_train_normalized), DataFrame(_y_test_normalized)

    def rename_columns(self, X_train: DataFrame = None, X_test: DataFrame = None):
        """
        Renames the columns of the feature datasets to a standardized format.
        Feature columns are renamed to 'feature_01', 'feature_02', etc.

        Args:
            X_train (DataFrame): Training features
            X_test (DataFrame): Test features

        Returns:
            Tuple of renamed X_train and X_test DataFrames
        """
        X_train_renamed = None
        X_test_renamed = None

        if X_train is not None:
            try:
                columns = list(X_train.columns)
                column_mapping = {
                    col: f"feature_{i + 1:02d}" for i, col in enumerate(columns)
                }

                X_train_renamed = X_train.rename(columns=column_mapping)
                log.info(
                    f"Renamed X_train columns: {list(X_train.columns)} -> {list(X_train_renamed.columns)}"
                )
            except Exception as e:
                log.error(f"Error renaming X_train columns: {e}")
                X_train_renamed = X_train

        if X_test is not None:
            try:
                columns = list(X_test.columns)
                column_mapping = {
                    col: f"feature_{i + 1:02d}" for i, col in enumerate(columns)
                }

                X_test_renamed = X_test.rename(columns=column_mapping)
                log.info(
                    f"Renamed X_test columns: {list(X_test.columns)} -> {list(X_test_renamed.columns)}"
                )
            except Exception as e:
                log.error(f"Error renaming X_test columns: {e}")
                X_test_renamed = X_test

        return X_train_renamed, X_test_renamed

    @property
    def preprocess_config(self) -> dict:
        return self._preprocess_config

    @preprocess_config.setter
    def preprocess_config(self, value: dict):
        if value is not None and not isinstance(value, dict):
            raise ValueError("preprocess_config must be a dictionary or None")
        self._preprocess_config = value
