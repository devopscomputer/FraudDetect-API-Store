import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from typing import List, Tuple

# Load the model
model_file = os.path.join(os.path.dirname(__file__), "ml/fraud_model.joblib")

try:
    loaded_model: Tuple[Pipeline, List[str]] = joblib.load(model_file)
    model, targets = loaded_model
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Load test data
try:
    x_test = pd.read_csv('data/test_data.csv')  # Ensure you have a test dataset
except Exception as e:
    raise RuntimeError(f"Failed to load test data: {e}")

# Run a prediction
try:
    p = model.predict([x_test.iloc[0]])

    if p[0] == 1:
        print("Fraud")
    else:
        print("Not Fraud")
except Exception as e:
    raise RuntimeError(f"Prediction failed: {e}")