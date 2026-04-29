# Loop de treinamento MLP com early stopping na validação.
import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from src.config import MODELS_DIR, SEED, set_global_seed
from src.models.models import ChurnMLPv2

logger = logging.getLogger(__name__)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_cls: type = ChurnMLPv2,
    epochs: int = 300,
    patience: int = 20,
    lr: float = 0.001,
    batch_size: int = 32,
    seed: int = SEED,
    save_path: str | None = None,
) -> tuple[nn.Module, dict[str, float]]:
    # Treina MLP com early stopping na val_loss.
    # Retorna (modelo treinado com melhor checkpoint, dict de métricas).

    set_global_seed(seed)

    # Conversão para tensores
    X_train_t = torch.FloatTensor(X_train)
    X_val_t = torch.FloatTensor(X_val)
    y_train_t = torch.FloatTensor(y_train)
    y_val_t = torch.FloatTensor(y_val)

    # DataLoader para mini-batches
    dataset = TensorDataset(X_train_t, y_train_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Modelo e otimizador
    input_dim = X_train_t.shape[1]
    model = model_cls(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Loss com peso para classe desbalanceada
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    pos_weight = torch.tensor([n_neg / n_pos])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    logger.info("Arquitetura %s: %d parâmetros", model_cls.__name__,
                sum(p.numel() for p in model.parameters()))
    logger.info("pos_weight: %.2f | lr: %s | batch_size: %d | patience: %d",
                pos_weight.item(), lr, batch_size, patience)

    # Caminho para salvar o melhor checkpoint
    save_path = save_path or str(MODELS_DIR / "mlp_best.pt")

    # ================================
    # LOOP DE TREINAMENTO
    # ================================
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # Fase de treino
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(dataloader)

        # Fase de validação
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t).squeeze()
            val_loss = criterion(val_outputs, y_val_t).item()

        # Early stopping baseado na loss de validação
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info("Early stopping na época %d | val_loss: %.4f",
                        epoch + 1, val_loss)
            break

        if (epoch + 1) % 10 == 0:
            logger.info("Época %d | Loss treino: %.4f | Loss val: %.4f",
                        epoch + 1, avg_train_loss, val_loss)

    # Recarrega melhor checkpoint
    model.load_state_dict(
        torch.load(save_path, weights_only=True, map_location="cpu")
    )
    model.eval()

    # ================================
    # AVALIAÇÃO NO CONJUNTO DE VALIDAÇÃO
    # ================================
    metrics = evaluate_model(model, X_val, y_val)

    logger.info(
        "Resultado final — AUC: %.3f | PR-AUC: %.3f | F1: %.3f | "
        "Precision: %.3f | Recall: %.3f",
        metrics["auc_roc"], metrics["pr_auc"], metrics["f1"],
        metrics["precision"], metrics["recall"],
    )

    return model, metrics


def evaluate_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    # Avalia modelo MLP e retorna dict com 5 métricas.
    model.eval()
    X_t = torch.FloatTensor(X)

    with torch.no_grad():
        logits = model(X_t).squeeze()
        probs = torch.sigmoid(logits).numpy()

    preds_binary = (probs >= threshold).astype(int)

    return {
        "auc_roc": float(roc_auc_score(y, probs)),
        "pr_auc": float(average_precision_score(y, probs)),
        "f1": float(f1_score(y, preds_binary)),
        "precision": float(precision_score(y, preds_binary, zero_division=0)),
        "recall": float(recall_score(y, preds_binary)),
    }
