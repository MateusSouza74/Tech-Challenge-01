# ================================
# IMPORTS
# ================================
import torch
import torch.nn as nn
import logging
from fastapi import FastAPI
from pydantic import BaseModel


# ================================
# CONFIGURAÇÃO DO LOGGER
# ================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================
# DEFINIÇÃO DO MODELO
# ================================
class ChurnMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


# ================================
# APLICAÇÃO FASTAPI
# ================================
app = FastAPI(title="Telco Churn API", version="1.0.0")


# ================================
# SCHEMA DE ENTRADA (PYDANTIC)
# ================================
class CustomerData(BaseModel):
    tenure_months: float
    monthly_charges: float
    total_charges: float
    contract: str
    payment_method: str
    internet_service: str
    online_security: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str


# ================================
# ENDPOINTS
# ================================
@app.get("/health")
def health():
    logger.info("Health check chamado")
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: CustomerData):
    logger.info(f"Previsão solicitada — tenure: {customer.tenure_months} meses")

    churn_probability = 0.42

    return {
        "churn_probability": churn_probability,
        "churn_prediction": churn_probability >= 0.5,
    }