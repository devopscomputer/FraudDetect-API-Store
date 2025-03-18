import pandas as pd

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria novas features a partir da interação entre variáveis existentes.
    
    :param df: DataFrame com dados de entrada
    :return: DataFrame com novas features
    """
    df['amount_time_interaction'] = df['Amount'] * df['V1']  # Exemplo de interação
    return df

def log_transform_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Aplica transformação logarítmica a features selecionadas.
    
    :param df: DataFrame com dados de entrada
    :param features: Lista de nomes de features a serem transformadas
    :return: DataFrame com features transformadas
    """
    for feature in features:
        df[feature] = df[feature].apply(lambda x: np.log(x + 1))  # +1 para evitar log(0)
    return df