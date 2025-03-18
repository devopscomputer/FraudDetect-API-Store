import numpy as np
from pydantic import BaseModel
from typing import List, Optional, Tuple
from sklearn.pipeline import Pipeline
import os
import joblib

class PredictionInput(BaseModel):
    data: List[float]  # Define input data type (should contain 29 features)

class PredictionOutput(BaseModel):
    category: int  # Define output category

class FraudDetector:
    model: Optional[Pipeline]
    targets: Optional[List[int]]

    def load_model(self):
        """Loads the model"""
        model_file = os.path.join(os.path.dirname(__file__), "ml/fraud_model.joblib")
        try:
            loaded_model: Tuple[Pipeline, List[int]] = joblib.load(model_file)
            self.model, self.targets = loaded_model
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        """Runs a prediction"""
        input_data_np = np.array(input_data.data).reshape(1, -1)

        # Validate input data shape
        if input_data_np.shape[1] != 29:  # Ensure that the input has 29 features
            raise ValueError(f"Expected 29 features, but got {input_data_np.shape[1]}")

        try:
            prediction = self.model.predict(input_data_np)
            category = self.targets[prediction[0]]
            return PredictionOutput(category=category)
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")