import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import StratifiedShuffleSplit

from src.utils.logger import log


class DatasetLoader:
    """Loads datasets from various sources and prepares them for experimentation.

    Supports loading from CSV files, sklearn built-in datasets (20 Newsgroups),
    and custom datasets. Handles data splitting, duplicate removal, and provides
    stratified k-fold splits for cross-validation.
    """

    def __init__(
        self,
        dataset_name: str = None,
        dataset_path: str = None,
        data_already_split: bool = False,
        train_data_path: str = None,
        test_data_path: str = None,
        random_seed: int = 1,
        data_split_ratio: float = 0.2,
        encoding: str = None,
        delimiter: str = None,
        target_column_name: str = None,
    ) -> None:

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.data_already_split = data_already_split
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.random_seed = random_seed
        self.data_split_ratio = data_split_ratio
        self.encoding = encoding
        self.delimiter = delimiter
        self.target_column_name = target_column_name

        self.X = None
        self.y = None

        log.info(f"-------------- {dataset_name.upper()} --------------")
        log.info(
            f"-------------- Initializing DatasetLoader for dataset: {dataset_name}"
        )

        if self.dataset_path is not None and not self.dataset_path == "":
            self.X, self.y = self.load_data()
        else:
            if self.dataset_name == "20_Newsgroups":
                self.X, self.y = self.load_20_Newsgroups_dataset()
            elif self.dataset_name == "Pt_Tweets_SA":
                self.X, self.y = self.load_Pt_Tweets_SA_dataset()

    def load_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Load dataset from CSV file(s), reuniting train/test splits if needed.

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if self.data_already_split:
            try:
                _train_data = pd.read_csv(
                    self.train_data_path,
                    encoding=self.encoding,
                    delimiter=self.delimiter,
                )
                _test_data = pd.read_csv(
                    self.test_data_path,
                    encoding=self.encoding,
                    delimiter=self.delimiter,
                )

                _data = pd.concat([_train_data, _test_data], ignore_index=True)
                # Remove duplicadas antes de qualquer divisão
                _data = self._apply_remove_duplicates(_data)
                log.info(
                    f"-------------- Reunited split dataset from {self.dataset_name} (train: {len(_train_data)}, test: {len(_test_data)}, total: {len(_data)})"
                )

                X = _data.drop(columns=[self.target_column_name])
                y = _data[self.target_column_name]

                log.info(
                    f"-------------- Successfully loaded and reunited dataset from {self.dataset_name}"
                )
                return X, y

            except Exception as e:
                log.error(
                    f"Error loading and reuniting split dataset from {self.dataset_name}:  {e}"
                )
                return None, None
        else:
            try:
                _data = pd.read_csv(
                    self.dataset_path, encoding=self.encoding, delimiter=self.delimiter
                )
                # Remove duplicadas antes de qualquer divisão
                _data = self._apply_remove_duplicates(_data)

                X = _data.drop(columns=[self.target_column_name])
                y = _data[self.target_column_name]

                log.info(
                    f"-------------- Successfully loaded dataset from {self.dataset_name} (total: {len(_data)})"
                )
                return X, y

            except Exception as e:
                log.error(f"Error loading dataset from {self.dataset_name}:  {e}")
                return None, None

    def load_20_Newsgroups_dataset(self) -> tuple[pd.DataFrame, pd.Series]:
        """Load and reunite the 20 Newsgroups dataset from sklearn.

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        try:
            _categories = [
                "alt.atheism",
                "talk.religion.misc",
                "comp.graphics",
                "sci.space",
            ]

            _train_data = fetch_20newsgroups(
                subset="train",
                categories=_categories,
                shuffle=True,
                random_state=self.random_seed,
            )

            _test_data = fetch_20newsgroups(
                subset="test",
                categories=_categories,
                shuffle=True,
                random_state=self.random_seed,
            )

            _combined_data = _train_data.data + _test_data.data
            _combined_target = list(_train_data.target) + list(_test_data.target)

            log.info(
                f"-------------- Reunited 20 Newsgroups dataset (train: {len(_train_data.data)}, test: {len(_test_data.data)}, total: {len(_combined_data)})"
            )
            log.info(
                f"-------------- Using full dataset with {len(_combined_data)} samples"
            )

            X = pd.DataFrame({"data": _combined_data})
            y = pd.Series(_combined_target, name="target")

            log.info(
                "-------------- Successfully loaded and sampled 20 Newsgroups dataset"
            )
            return X, y

        except Exception as e:
            log.error(f"Error loading 20 Newsgroups dataset: {e}")
            return None, None

    def load_Pt_Tweets_SA_dataset(self) -> tuple[pd.DataFrame, pd.Series]:
        """Load and reunite the Portuguese Tweets Sentiment Analysis dataset.

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        try:
            _train_data = pd.read_csv(
                self.train_data_path, delimiter=self.delimiter, encoding="latin-1"
            )
            _test_data = pd.read_csv(
                self.test_data_path, delimiter=self.delimiter, encoding="latin-1"
            )

            _data = pd.concat([_train_data, _test_data], ignore_index=True)
            # Remove duplicadas antes de qualquer divisão
            _data = self._apply_remove_duplicates(_data)
            log.info(
                f"-------------- Reunited Pt_Tweets_SA dataset (train: {len(_train_data)}, test: {len(_test_data)}, total: {len(_data)})"
            )

            X = _data.drop(columns=[self.target_column_name])
            y = _data[self.target_column_name]

            log.info(
                "-------------- Successfully loaded and reunited Pt_Tweets_SA dataset"
            )
            return X, y

        except Exception as e:
            log.error(f"Error loading Pt_Tweets_SA dataset: {e}")
            return None, None

    def stratified_k_splits(self, n_splits: int = 10) -> list[tuple]:
        """Execute stratified k-fold splits maintaining class proportions.

        Args:
            n_splits: Number of train/test splits to generate

        Returns:
            List of (train_indices, test_indices) tuples
        """
        if self.X is None or self.y is None:
            log.error("Data not loaded. Cannot perform stratified split.")
            return []

        try:
            sss = StratifiedShuffleSplit(
                n_splits=n_splits, test_size=self.data_split_ratio, random_state=1
            )
            splits = []

            for train_index, test_index in sss.split(self.X, self.y):
                splits.append((train_index, test_index))

        except Exception as e:
            log.error(f"Error during stratified split: {e}")
            return []

        log.info(
            f"-------------- Completed {n_splits} stratified train-test splits ({1 - self.data_split_ratio}/{self.data_split_ratio})"
        )
        log.info("Splits details:")
        for i, (train_idx, test_idx) in enumerate(splits):
            log.info(
                f"  Split {i + 1}: Train indices count: {len(train_idx)}, Test indices count: {len(test_idx)}"
            )
        return splits

    def _apply_remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove linhas duplicadas do DataFrame e retorna o DataFrame limpo.

        Aplicado antes de qualquer divisão treino/teste para que X/y reflitam o dataset sem duplicatas.
        """
        try:
            initial_count = len(df)
            df = df.drop_duplicates().reset_index(drop=True)
            final_count = len(df)
            removed = initial_count - final_count
            if removed > 0:
                log.info(
                    f"Removed {removed} duplicate rows (from {initial_count} to {final_count})"
                )
            else:
                log.info("No duplicate rows found")
            return df
        except Exception as e:
            log.error(f"Error while removing duplicates: {e}")
            return df

    @property
    def encoding(self):
        return self._encoding

    @property
    def delimiter(self):
        return self._delimiter

    @encoding.setter
    def encoding(self, value: str):
        if value is None or value == "":
            self._encoding = "utf-8"
        else:
            self._encoding = value

    @delimiter.setter
    def delimiter(self, value: str):
        if value is None or value == "":
            self._delimiter = ","
        else:
            self._delimiter = value
