import unittest
import pandas as pd
from ml.feature_engineering import create_interaction_features, log_transform_features

class TestFeatureEngineering(unittest.TestCase):
    
    def setUp(self):
        self.df = pd.DataFrame({
            'V1': [0.1, 0.2],
            'Amount': [100, 200]
        })

    def test_create_interaction_features(self):
        df_with_interactions = create_interaction_features(self.df)
        self.assertIn('amount_time_interaction', df_with_interactions.columns)

    def test_log_transform_features(self):
        df_transformed = log_transform_features(self.df, ['Amount'])
        self.assertTrue((df_transformed['Amount'] > 0).all())  # Verifica se todos os valores são positivos

if __name__ == '__main__':
    unittest.main()