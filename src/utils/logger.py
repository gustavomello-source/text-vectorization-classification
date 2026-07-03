import logging as log
import os
import shutil
from datetime import datetime

import pandas as pd


def setup_logging():
    if not os.path.exists("logs"):
        os.makedirs("logs")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_folder = f"logs/experiment_{timestamp}"

    # Create experiment folder
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)

    log_filename = f"{experiment_folder}/experiment.log"

    log.basicConfig(
        level=log.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[log.FileHandler(log_filename), log.StreamHandler()],
    )

    # Return experiment folder path for saving configs
    return experiment_folder


def save_configs(experiment_folder, config_files=None):
    """Save configuration files used in the experiment"""
    if config_files is None:
        config_files = [
            "config/dataset_config.ini",
            "config/preprocess_config.ini",
            "config/vectorizers_config.ini",
        ]

    configs_folder = f"{experiment_folder}/configs"
    if not os.path.exists(configs_folder):
        os.makedirs(configs_folder)

    log.info(f"-------------- Saving configuration files to {configs_folder}")

    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                shutil.copy2(config_file, configs_folder)
                log.info(f"Saved config: {config_file}")
            except Exception as e:
                log.error(f"Error saving config {config_file}: {e}")
        else:
            log.warning(f"Config file not found: {config_file}")

    log.info("-------------- Configuration files saved successfully")


def setup_results_csv(experiment_folder):
    """Set up a CSV file to store experiment results"""
    results_file = f"{experiment_folder}/experiment_results.csv"

    columns = [
        "dataset_name",
        "model_name",
        "fold_number",
        "vectorizer_name",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "confusion_matrix",
        "classification_report",
        "train_time_seconds",
        "prediction_time_seconds",
        "total_samples",
        "train_samples",
        "test_samples",
        "timestamp",
    ]

    results_df = pd.DataFrame(columns=columns)

    try:
        results_df.to_csv(results_file, index=False, sep=";")
        log.info(f"-------------- Results CSV created at: {results_file}")
        log.info(f"CSV structure: {'; '.join(columns)}")
        return results_file
    except Exception as e:
        log.error(f"Error creating results CSV: {e}")
        return None


def append_results_to_csv(
    results_file,
    dataset_name,
    model_name,
    fold_number,
    vectorizer_name,
    accuracy=None,
    precision=None,
    recall=None,
    f1_score=None,
    confusion_matrix=None,
    classification_report=None,
    total_vectorization_time=None,
    train_time=None,
    prediction_time=None,
    total_samples=None,
    train_samples=None,
    test_samples=None,
):
    """Append a new result row to the experiment results CSV"""
    try:
        # Formatação da matriz de confusão para evitar quebras de linha no CSV
        if confusion_matrix is not None:
            # Converte a matriz de confusão para uma string sem quebras de linha
            cm_str = (
                str(confusion_matrix.tolist()).replace("\n", " ").replace("  ", " ")
            )
        else:
            cm_str = None

        # Formatação do classification report para evitar quebras de linha no CSV
        if classification_report is not None:
            cr_str = str(classification_report).replace("\n", " | ").replace("  ", " ")
        else:
            cr_str = None

        new_row = {
            "dataset_name": dataset_name,
            "model_name": model_name,
            "fold_number": fold_number,
            "vectorizer_name": vectorizer_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "confusion_matrix": cm_str,
            "classification_report": cr_str,
            "total_vectorization_time": total_vectorization_time,
            "train_time_seconds": train_time,
            "prediction_time_seconds": prediction_time,
            "total_samples": total_samples,
            "train_samples": train_samples,
            "test_samples": test_samples,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        if os.path.exists(results_file):
            df = pd.read_csv(results_file, sep=";")
        else:
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            df = pd.DataFrame()

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        df.to_csv(results_file, index=False, sep=";")

        log.info(
            f"Results appended: {dataset_name} | {model_name} | Fold {fold_number} | {vectorizer_name}"
        )
        return True

    except Exception as e:
        log.error(f"Error appending results to CSV: {e}")
        return False
