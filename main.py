from fastapi import FastAPI, HTTPException, Depends
from typing import List
from fastapi.security import APIKeyHeader
from prediction_endpoint import FraudDetector, PredictionOutput, PredictionInput
from ml.user_behavior import UserBehavior
from ml.alerts import AlertService
from ml.dashboard import Dashboard
from ml.data_preprocessing import preprocess_data
from ml.feature_engineering import create_interaction_features
from ml.hyperparameter_tuning import tune_hyperparameters
from pydantic import BaseModel, Field, validator
from datetime import datetime
import pandas as pd

# Definição do modelo AlertInput
class AlertInput(BaseModel):
    message: str

# Definição do modelo DashboardReport
class DashboardReport(BaseModel):
    total_frauds: int
    total_transactions: int
    fraud_rate: float

# Definição do modelo PredictionInput com validações
class PredictionInput(BaseModel):
    user_id: int
    amount: float = Field(..., gt=0, description="Valor da transação deve ser maior que zero.")
    product_id: int
    transaction_type: str
    timestamp: datetime

    @validator('amount')
    def check_amount(cls, v):
        if v > 10000:  # Exemplo de limite
            raise ValueError("Valor da transação não pode ser superior a 10.000.")
        return v

    @validator('timestamp')
    def check_time(cls, v):
        if v.hour < 6 or v.hour > 22:  # Exemplo de horário incomum
            raise ValueError("Transações só podem ocorrer entre 6h e 22h.")
        return v

# Configuração da API
app = FastAPI()

# Configuração da autenticação via API Key
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME)

async def authenticate(api_key: str = Depends(api_key_header)):
    if api_key != "YOUR_SECRET_API_KEY":  # Substitua por uma verificação real
        raise HTTPException(status_code=403, detail="Unauthorized")

# Inicializando os serviços
fraud_model = FraudDetector()
user_behavior_service = UserBehavior()
alert_service = AlertService()
dashboard_service = Dashboard()

@app.on_event("startup")
async def startup():
    try:
        fraud_model.load_model()
        print("Model loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load model during startup: {e}")

@app.post("/v1/prediction", response_model=dict)
async def prediction(input_data: PredictionInput, api_key: str = Depends(authenticate)):
    df = pd.DataFrame([input_data.dict()])
    df = preprocess_data(df)
    df = create_interaction_features(df)
    output = fraud_model.predict(df)

    # Supondo que o modelo retorna uma pontuação de fraude
    fraud_score = output['fraud_score']  # Exemplo de como obter a pontuação
    reason = "Valor da transação é incomum para o perfil do usuário."  # Exemplo de razão

    return {
        "is_fraudulent": output['is_fraudulent'],
        "fraud_score": fraud_score,
        "reason": reason
    }

@app.post("/v1/user-behavior", response_model=dict)
async def analyze_user_behavior(user_data: List[float], api_key: str = Depends(authenticate)):
    try:
        analysis = user_behavior_service.analyze(user_data)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/v1/alerts", response_model=dict)
async def send_alert(alert_input: AlertInput, api_key: str = Depends(authenticate)):
    try:
        alert_service.send_alert(alert_input.message)
        return {"status": "Alert sent successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/dashboard", response_model=DashboardReport)
async def get_dashboard_report(api_key: str = Depends(authenticate)):
    try:
        report = dashboard_service.generate_report()
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/tune-hyperparameters", response_model=dict)
async def tune_model(api_key: str = Depends(authenticate)):
    try:
        df = pd.read_csv('path_to_your_dataset.csv')  # Ajuste o caminho para o seu dataset
        X, y = preprocess_data(df)
        best_model = tune_hyperparameters(X, y)
        return {"status": "Hyperparameters tuned successfully", "model": str(best_model)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))