from fastapi import FastAPI, HTTPException, Depends
from typing import List
from fastapi.security import APIKeyHeader
from prediction_endpoint import FraudDetector, PredictionOutput, PredictionInput
from ml.user_behavior import UserBehavior
from ml.alerts import AlertService
from ml.dashboard import Dashboard

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

@app.post("/v1/prediction", response_model=PredictionOutput)
async def prediction(input_data: PredictionInput, api_key: str = Depends(authenticate)) -> PredictionOutput:
    output = fraud_model.predict(input_data)
    return output

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

# Novo endpoint de teste de fraude
@app.post("/v1/test-fraud-detection", response_model=PredictionOutput)
async def test_fraud_detection(input_data: PredictionInput, api_key: str = Depends(authenticate)):
    """
    Endpoint para testar a detecção de fraudes com dados fictícios.

    - **user_id**: ID do usuário.
    - **amount**: Valor da compra.
    - **product_id**: ID do produto.
    - **transaction_type**: Tipo de transação.
    - **timestamp**: Data e hora da compra.

    **Exemplo de Request Body**:
    ```json
    {
        "user_id": 123,
        "amount": 150.75,
        "product_id": 456,
        "transaction_type": "purchase",
        "timestamp": "2023-03-18T12:00:00"
    }
    ```

    **Resposta**:
    - **200 OK**: Retorna um objeto que indica se a compra é fraudulenta.
    """
    if input_data.amount > 100:  # Exemplo de regra fictícia
        return PredictionOutput(is_fraudulent=True)
    return PredictionOutput(is_fraudulent=False)