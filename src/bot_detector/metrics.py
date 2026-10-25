import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true, y_pred_proba, threshold: float = 0.5
) -> dict[str, float]:
    """Logloss + threshold-based classification metrics for MLflow logging."""
    y_pred_proba = np.clip(np.asarray(y_pred_proba, dtype=float), 1e-7, 1 - 1e-7)
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        "logloss": float(log_loss(y_true, y_pred_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }