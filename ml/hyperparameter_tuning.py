from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

def tune_hyperparameters(X, y):
    """
    Realiza tuning de hiperparâmetros usando Grid Search.
    
    :param X: Features do DataFrame
    :param y: Labels do DataFrame
    :return: Melhor modelo encontrado
    """
    model = LogisticRegression()
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'saga']
    }
    
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_