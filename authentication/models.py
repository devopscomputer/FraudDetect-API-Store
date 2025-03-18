from pydantic import BaseModel
from typing import List, Optional

class PredictionInput(BaseModel):
    user_id: int
    amount: float
    product_id: int
    transaction_type: str
    timestamp: str  # Pode ser ajustado para datetime se preferir

class PredictionOutput(BaseModel):
    is_fraudulent: bool

class UserBehaviorInput(BaseModel):
    user_id: int
    actions: List[str]  # Lista de ações que o usuário realizou

class UserBehaviorOutput(BaseModel):
    behavior_analysis: str  # Resumo da análise do comportamento

class AlertInput(BaseModel):
    message: str

class AlertOutput(BaseModel):
    status: str

class DashboardReport(BaseModel):
    total_transactions: int
    fraudulent_transactions: int
    non_fraudulent_transactions: int
    fraud_rate: float