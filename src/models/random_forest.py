"""
Random Forest model strategy
"""
import time
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from src.utils.logger import log
from .base_strategy import ModelStrategy

class RandomForestStrategy(ModelStrategy):
    def train(self, X_train: DataFrame, y_train: DataFrame):
        """Train Random Forest model"""
        try:
            log.info("Training Random Forest model")
            model = RandomForestClassifier(random_state=1)
            
            # Handle DataFrame to array conversion
            X_array = X_train.values if isinstance(X_train, DataFrame) else X_train
            y_array = y_train.values.ravel() if isinstance(y_train, DataFrame) else y_train.ravel()
            
            train_time = time.perf_counter()
            model.fit(X_array, y_array)
            train_time = time.perf_counter() - train_time
            log.info("Random Forest training completed")
            return model, train_time
            
        except Exception as e:
            log.error(f"Error training Random Forest: {e}")
            return None, None
    
    def predict(self, model, X_test: DataFrame):
        """Make predictions with Random Forest"""
        try:
            X_array = X_test.values if isinstance(X_test, DataFrame) else X_test
            prediction_time = time.perf_counter()
            predictions = model.predict(X_array)
            prediction_time = time.perf_counter() - prediction_time
            probabilities = model.predict_proba(X_array)
            
            log.info("Random Forest prediction completed")
            return predictions, probabilities, prediction_time
            
        except Exception as e:
            log.error(f"Error in Random Forest prediction: {e}")
            return None, None, None