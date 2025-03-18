import unittest
import pandas as pd
from ml.model_train import train_model  # Supondo que você tenha uma função train_model

class TestModel(unittest.TestCase):
    
    def setUp(self):
        # Configuração para os testes
        self.df = pd.DataFrame({
            'V1': [0.1, 0.2],
            'V2': [0.4, 0.5],
            'Amount': [100, 200],
            'Class': [0, 1]
        })
        self.model = train_model(self.df)  # Treine o modelo aqui

    def test_model_prediction(self):
        prediction = self.model.predict([[0.1, 0.4, 100]])  # Exemplo de predição
        self.assertIn(prediction, [0, 1])  # Espera-se que a previsão seja 0 ou 1

if __name__ == '__main__':
    unittest.main()