import logging as log

from src.config.config_reader import ConfigReader
from src.data.dataset_loader import DatasetLoader
from src.models.models import MLModel
from src.preprocessing.preprocess import Preprocessor
from src.utils.logger import append_results_to_csv, save_configs, setup_logging
from src.utils.results import summarize_results_in_all_csvs
from src.vectorizers.vectorizer import TextVectorizer


def main():
    experiment_folder = setup_logging()

    ConfigReader.setup_cuda_environment()

    config_files = [
        "config/dataset_config.ini",
        "config/preprocess_config.ini",
        "config/vectorizers_config.ini",
    ]
    config, preprocess_config, vectorizers_config = ConfigReader.read_config_files(
        config_files
    )

    save_configs(experiment_folder, config_files)

    dataset_loader_config = config["DatasetLoader Config"]

    datasets_dict = {
        "20_Newsgroups": "External Dataset",
        "IMDB_Reviews": config["IMDB_Reviews"],
    }

    vectorizers_dict = {
        "BoW": {},
        "TF-IDF": {},
        "Word2Vec": {},
        "GloVe": {},
        "Doc2Vec": {},
        "BERT": {},
        "RoBERTa": {},
        "ALBERT": {},
        "E5": {},
        "InstructorEmbedding": {},
        "GPT2": {},
    }

    models_dict = {
        "LinearSVC": {},
        "LogisticRegression": {},
        "RandomForest": {},
    }

    # Loop para cada dataset
    for dataset_name, dataset_config in datasets_dict.items():
        if dataset_name == "20_Newsgroups":
            dataset = DatasetLoader(
                dataset_name="20_Newsgroups",
                random_seed=dataset_loader_config.getint("random_seed"),
                data_split_ratio=dataset_loader_config.getfloat("data_split_ratio"),
            )

        elif dataset_config.get("dataset_name") == dataset_name:
            dataset = DatasetLoader(
                dataset_name=dataset_config.get("dataset_name"),
                dataset_path=dataset_config.get("dataset_path"),
                data_already_split=dataset_config.getboolean("data_already_split"),
                train_data_path=dataset_config.get("train_data_path"),
                test_data_path=dataset_config.get("test_data_path"),
                random_seed=dataset_loader_config.getint("random_seed"),
                data_split_ratio=dataset_loader_config.getfloat("data_split_ratio"),
                encoding=dataset_config.get("encoding"),
                delimiter=dataset_config.get("delimiter"),
                target_column_name=dataset_config.get("target_column_name"),
            )

        else:
            log.warning("No valid dataset configuration found for %s", dataset_name)
            break

        folds = dataset.stratified_k_splits(n_splits=10)

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            log.info(
                f"================ Starting Fold {fold_idx + 1}/{len(folds)} for Dataset: {dataset_name} ================="
            )

            X_train, X_test = dataset.X.iloc[train_idx], dataset.X.iloc[test_idx]
            y_train, y_test = dataset.y.iloc[train_idx], dataset.y.iloc[test_idx]

            full_preprocessor = Preprocessor(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                preprocess_config=preprocess_config[dataset_name],
                encoding=dataset.encoding,
            )

            X_train, y_train, X_test, y_test = full_preprocessor.preprocess_data()

            for vectorizer_name, vectorizer_config in vectorizers_dict.items():
                vectorizer = TextVectorizer(
                    method=vectorizer_name, vectorizer_params=vectorizer_config
                )
                X_train_vectorized, fitted_vectorizer, vectorizer_train_time = (
                    vectorizer.vectorize_data(X_train)
                )
                X_test_vectorized, vectorizer_transform_time = (
                    vectorizer.transform_data(fitted_vectorizer, X_test)
                )
                total_vectorization_time = (
                    f"{vectorizer_train_time + vectorizer_transform_time:.4f}"
                )

                for model_name, model_config in models_dict.items():
                    model = MLModel(method=model_name, model_params=model_config)
                    fitted_model, train_time = model.train_model(
                        X_train_vectorized, y_train
                    )
                    evaluation = model.evaluate(X_test_vectorized, y_test)

                    log.info(
                        f"Model: {model_name} on Dataset: {dataset_name} with Vectorizer: {vectorizer_name}"
                    )
                    append_results_to_csv(
                        experiment_folder
                        + "/results/"
                        + f"{dataset_name}/{model_name}/{vectorizer_name}.csv",
                        dataset_name,
                        model_name,
                        fold_idx + 1,
                        vectorizer_name,
                        accuracy=evaluation["classification_report"]["accuracy"],
                        precision=evaluation["classification_report"]["macro avg"][
                            "precision"
                        ],
                        recall=evaluation["classification_report"]["macro avg"][
                            "recall"
                        ],
                        f1_score=evaluation["classification_report"]["macro avg"][
                            "f1-score"
                        ],
                        confusion_matrix=evaluation["confusion_matrix"],
                        classification_report=evaluation["classification_report"],
                        total_vectorization_time=total_vectorization_time,
                        train_time=f"{train_time:.4f}",
                        prediction_time=f"{evaluation['prediction_time']:.4f}",
                        total_samples=len(X_train) + len(X_test),
                        train_samples=len(X_train),
                        test_samples=len(X_test),
                    )

                vectorizer.cleanup()

    summarize_results_in_all_csvs(results_root=experiment_folder + "/results")


if __name__ == "__main__":
    main()
