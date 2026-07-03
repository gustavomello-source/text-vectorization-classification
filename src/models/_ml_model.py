"""
Main ML model class that orchestrates different model strategies using the Strategy pattern.

Supports multiple classification models: LinearSVC, LogisticRegression, and RandomForest.
Handles training, prediction, and evaluation with proper timing and error handling.
"""

from pandas import DataFrame
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.utils.logger import log

from .linear_svc import LinearSVCStrategy
from .logistic_regression import LogisticRegressionStrategy
from .random_forest import RandomForestStrategy


class MLModel:
    def __init__(self, method: str, model_params: dict = None):
        self._strategies = {
            "LinearSVC": LinearSVCStrategy(),
            "LogisticRegression": LogisticRegressionStrategy(),
            "RandomForest": RandomForestStrategy(),
        }

        self._model_params = model_params

        if method not in self._strategies:
            available_methods = list(self._strategies.keys())
            raise ValueError(
                f"Unsupported method '{method}'. Available methods: {available_methods}"
            )

        self._method = method
        self._fitted_model = None
        self.train_time = None
        self.prediction_time = None

        log.info(f"-------------- Initializing ML model: {method}")

    def train_model(self, X_train: DataFrame, y_train: DataFrame) -> tuple[any, float]:
        """Train the model using the selected strategy"""
        log.info(f"Training model using method: {self.method}")

        try:
            strategy = self._strategies[self.method]
            self._fitted_model, self.train_time = strategy.train(X_train, y_train)

            if self._fitted_model is not None:
                log.info(
                    f"Model training complete for {self.method} in {self.train_time:.4f} seconds"
                )
                return self._fitted_model, self.train_time
            else:
                log.warning(f"Model training returned None for method: {self.method}")
                return None, None

        except Exception as e:
            log.error(f"Error during {self.method} training: {e}")
            return None, None

    def predict(self, X_test: DataFrame) -> tuple[any, any, float]:
        """Make predictions using the fitted model"""
        if self._fitted_model is None:
            log.error("Model has not been trained yet. Call train_model() first.")
            return None, None, None

        log.info(f"Making predictions using {self.method}")

        try:
            strategy = self._strategies[self.method]
            predictions, probabilities, prediction_time = strategy.predict(
                self._fitted_model, X_test
            )

            if predictions is not None:
                log.info(
                    f"Prediction complete for {self.method} in {prediction_time:.4f} seconds"
                )
                return predictions, probabilities, prediction_time
            else:
                log.warning(f"Prediction returned None for method: {self.method}")
                return None, None, None

        except Exception as e:
            log.error(f"Error during {self.method} prediction: {e}")
            return None, None, None

    def evaluate(self, X_test: DataFrame, y_test: DataFrame) -> dict:
        """Evaluate the model performance"""
        predictions, probabilities, prediction_time = self.predict(X_test)

        if predictions is None:
            log.error("Cannot evaluate: Prediction failed")
            return None

        try:
            y_true = (
                y_test.values.ravel()
                if isinstance(y_test, DataFrame)
                else y_test.ravel()
            )

            accuracy = accuracy_score(y_true, predictions)
            report = classification_report(y_true, predictions, output_dict=True)
            conf_matrix = confusion_matrix(y_true, predictions)

            log.info(f"Model Evaluation for {self.method}:")
            log.info(f"Accuracy: {accuracy:.4f}")
            log.info(f"Classification Report:\n{report}")
            log.info(f"Confusion Matrix:\n{conf_matrix}")

            return {
                "accuracy": accuracy,
                "classification_report": report,
                "confusion_matrix": conf_matrix,
                "predictions": predictions,
                "probabilities": probabilities,
                "prediction_time": prediction_time,
            }

        except Exception as e:
            log.error(f"Error during model evaluation: {e}")
            return None

    def get_available_methods(self) -> list[str]:
        """Return list of available ML methods"""
        return list(self._strategies.keys())

    @property
    def method(self) -> str:
        return self._method

    @method.setter
    def method(self, value: str) -> None:
        if not isinstance(value, str):
            raise ValueError("Method must be a string")
        if value not in self._strategies:
            available = list(self._strategies.keys())
            raise ValueError(f"Unsupported method '{value}'. Available: {available}")
        self._method = value
        self._fitted_model = None

    @property
    def fitted_model(self) -> any:
        """Return the fitted model (read-only)"""
        return self._fitted_model

    def is_trained(self) -> bool:
        """Check if model has been trained"""
        return self._fitted_model is not None
