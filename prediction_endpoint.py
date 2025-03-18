import numpy as np
from pydantic import BaseModel, Field
from typing import List
from sklearn.linear_model import LogisticRegression
import os
import joblib
from datetime import datetime
from fastapi import FastAPI, HTTPException

app = FastAPI()

class PredictionInput(BaseModel):
    user_id: int = Field(..., example=123)                     # ID do usuário
    amount: float = Field(..., ge=0, example=150.75)          # Valor da compra, deve ser não negativo
    product_id: int = Field(..., example=456)                  # ID do produto
    transaction_type: str = Field(..., example='purchase')     # Tipo de transação
    timestamp: datetime = Field(..., example='2023-03-18T12:00:00')  # Data e hora da compra

class PredictionOutput(BaseModel):
    is_fraudulent: bool = Field(..., example=False)           # True se a compra é fraudulenta

class FraudDetector:
    model: LogisticRegression

    def load_model(self):
        """Carrega o modelo de detecção de fraudes."""
        model_file = os.path.join(os.path.dirname(__file__), "ml/fraud_model.joblib")
        try:
            self.model = joblib.load(model_file)
            print("Modelo carregado com sucesso.")
        except Exception as e:
            raise RuntimeError(f"Falha ao carregar o modelo: {e}")

    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        """Executa uma predição para determinar se a compra é fraudulenta."""
        input_data_np = np.array([[input_data.user_id, input_data.amount]])  # Adapte conforme necessário

        expected_features = self.model.coef_.shape[1]  # Obtém o número de características do modelo
        if input_data_np.shape[1] != expected_features:
            raise ValueError(f"Esperado {expected_features} características, mas obteve {input_data_np.shape[1]}.")

        try:
            prediction = self.model.predict(input_data_np)
            is_fraudulent = bool(prediction[0])  # Converte para booleano
            return PredictionOutput(is_fraudulent=is_fraudulent)
        except Exception as e:
            raise RuntimeError(f"Predição falhou: {e}")

# Instancia do detector de fraudes
fraud_detector = FraudDetector()
fraud_detector.load_model()

@app.post("/prediction", response_model=PredictionOutput)
async def prediction(input_data: PredictionInput):
    """
    Endpoint para prever se uma compra é fraudulenta.

    - **user_id**: ID do usuário.
    - **amount**: Valor da compra (não negativo).
    - **product_id**: ID do produto.
    - **transaction_type**: Tipo de transação (ex: 'purchase').
    - **timestamp**: Data e hora da compra no formato ISO.
    
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

    **Respostas**:
    - **200 OK**: Retorna um objeto que indica se a compra é fraudulenta.
        **Exemplo de Resposta**:
        ```json
        {
            "is_fraudulent": false
        }
        ```
    - **422 Unprocessable Entity**: Retorna um erro de validação se os dados de entrada não forem válidos.
        **Exemplo de Resposta**:
        ```json
        {
            "detail": [
                {
                    "loc": ["body", "amount"],
                    "msg": "value is not a valid float",
                    "type": "value_error.float"
                }
            ]
        }
        ```

    Ao testar o endpoint:
    1. Clique em "Try it out".
    2. Preencha os campos com dados válidos.
    3. Clique em "Execute" para enviar a requisição.
    4. Verifique a resposta que aparece abaixo, observando se ela corresponde às expectativas.
    """
    return fraud_detector.predict(input_data)