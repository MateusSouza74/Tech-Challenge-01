"""Pipeline de pré-processamento para Telco Churn."""
import json
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DROP_COLS = [
    "CustomerID", "Country", "State", "City", "Lat Long",
    "Churn Reason", "Churn Score", "Churn Value", "CLTV",
]

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features (X) e alvo (y), aplica one-hot encoding."""
    X = df.drop(columns=DROP_COLS + ["Churn Label"])
    y = (df["Churn Label"] == "Yes").astype(int)
    X = pd.get_dummies(X)
    logger.info(
        "Features: %d colunas | Positivos: %d (%.1f%%)",
        X.shape[1], y.sum(), y.mean() * 100,
    )
    return X, y


def load_scaler(path: Path | None = None) -> StandardScaler:
    """Carrega o scaler treinado do disco."""
    path = path or MODELS_DIR / "scaler.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def load_feature_columns(path: Path | None = None) -> list[str]:
    """Carrega as colunas de features usadas no treino."""
    path = path or MODELS_DIR / "feature_columns.json"
    with open(path) as f:
        return json.load(f)


def preprocess_input(
    data: dict,
    scaler: StandardScaler,
    feature_columns: list[str],
) -> np.ndarray:
    """Transforma um dict de entrada no formato esperado pelo modelo.

    Pipeline: dict → DataFrame → one-hot → alinhamento de colunas → scaling.
    Colunas ausentes são preenchidas com 0; colunas extras são ignoradas.
    """
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_columns, fill_value=0)
    scaled = scaler.transform(df.values)
    return scaled
