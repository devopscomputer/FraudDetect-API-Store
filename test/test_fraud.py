from fastapi.testclient import TestClient
from main import app  # Certifique-se de que o app está acessível aqui
from prediction_endpoint import PredictionInput  # Importa o modelo de entrada

client = TestClient(app)

def test_fraud_detection_high_amount():
    # Dados de entrada para o teste
    input_data = {
        "user_id": 123,
        "amount": 150.75,
        "product_id": 456,
        "transaction_type": "purchase",
        "timestamp": "2023-03-18T12:00:00"
    }

    # Chama o endpoint de teste de fraude
    response = client.post("/test-fraud-detection", json=input_data)

    # Verifica se a resposta é 200 OK
    assert response.status_code == 200
    # Verifica se a transação é fraudulenta
    assert response.json()["is_fraudulent"] is True

def test_fraud_detection_low_amount():
    # Dados de entrada com valor baixo
    input_data = {
        "user_id": 124,
        "amount": 50.00,
        "product_id": 457,
        "transaction_type": "purchase",
        "timestamp": "2023-03-18T12:00:00"
    }

    # Chama o endpoint de teste de fraude
    response = client.post("/test-fraud-detection", json=input_data)

    # Verifica se a resposta é 200 OK
    assert response.status_code == 200
    # Verifica se a transação não é fraudulenta
    assert response.json()["is_fraudulent"] is False