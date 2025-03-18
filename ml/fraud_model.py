from sklearn.ensemble import RandomForestClassifier  # Exemplo de um modelo
import pandas as pd

class FraudDetector:
    def __init__(self):
        self.model = RandomForestClassifier()
        self.model_loaded = False

    def load_model(self):
        # Aqui você carregaria seu modelo treinado
        self.model_loaded = True

    def predict(self, df):
        if not self.model_loaded:
            raise RuntimeError("Model not loaded.")
        
        # Simulando predições
        prediction = self.model.predict(df)
        probability = self.model.predict_proba(df)[:, 1]  # Probabilidade da classe positiva

        return {"is_fraudulent": prediction[0], "fraud_score": probability[0]}