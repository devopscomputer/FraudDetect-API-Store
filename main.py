from fastapi import FastAPI
from prediction_endpoint import FraudDetector, PredictionOutput, PredictionInput

app = FastAPI()
fraud_model = FraudDetector()

@app.post("/prediction")
async def prediction(input_data: PredictionInput) -> PredictionOutput:
    output = fraud_model.predict(input_data)
    return output

@app.on_event("startup")
async def startup():
    try:
        fraud_model.load_model()
        print("Model loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load model during startup: {e}")