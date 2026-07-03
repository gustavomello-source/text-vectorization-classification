"""
Estratégia de modelo Linear SVC
"""
import time
import numpy as np
from pandas import DataFrame
from sklearn.svm import LinearSVC
from src.utils.logger import log
from .base_strategy import ModelStrategy

class LinearSVCStrategy(ModelStrategy):
    def train(self, X_train: DataFrame, y_train: DataFrame):
        """Treina modelo Linear SVC"""
        try:
            log.info("Training Linear SVC model")
            model = LinearSVC(random_state=1)
            
            # Converte DataFrame para array
            X_array = X_train.values if isinstance(X_train, DataFrame) else X_train
            y_array = y_train.values.ravel() if isinstance(y_train, DataFrame) else y_train.ravel()
            
            train_time = time.perf_counter()
            model.fit(X_array, y_array)
            train_time = time.perf_counter() - train_time
            log.info("Linear SVC training completed")
            return model, train_time

        except Exception as e:
            log.error(f"Error training Linear SVC: {e}")
            return None, None
    
    def predict(self, model, X_test: DataFrame):
        """Make predictions with Linear SVC"""
        try:
            X_array = X_test.values if isinstance(X_test, DataFrame) else X_test
            prediction_time = time.perf_counter()
            predictions = model.predict(X_array)
            prediction_time = time.perf_counter() - prediction_time

            try:
                decision_scores = model.decision_function(X_array)
                if decision_scores.ndim == 1:
                    probabilities = np.column_stack([1 - decision_scores, decision_scores])
                else:
                    exp_scores = np.exp(decision_scores)
                    probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            except:
                probabilities = None
                log.warning("Could not compute probabilities for Linear SVC")
            
            log.info("Linear SVC prediction completed")
            return predictions, probabilities, prediction_time
            
        except Exception as e:
            log.error(f"Error in Linear SVC prediction: {e}")
            return None, None, None