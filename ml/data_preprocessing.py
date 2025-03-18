import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza o pré-processamento dos dados, incluindo limpeza e normalização.
    
    :param df: DataFrame com dados brutos
    :return: DataFrame pré-processado
    """
    # Exemplo de limpeza
    df = df.dropna()  # Remove valores nulos
    # Normalização ou outras transformações podem ser adicionadas aqui
    return df

def balance_classes(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Aplica SMOTE para balancear as classes no conjunto de dados.
    
    :param X: Features do DataFrame
    :param y: Labels do DataFrame
    :return: Features e labels balanceados
    """
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def split_data(df: pd.DataFrame, target: str) -> tuple:
    """
    Divide os dados em conjuntos de treino e teste.
    
    :param df: DataFrame com dados
    :param target: Nome da coluna alvo
    :return: Conjuntos de treino e teste
    """
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)