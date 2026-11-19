import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_recall_fscore_support, roc_auc_score


def choose_threshold(y, probability):
    """Validation F1 only. Larger threshold wins an exact tie."""
    y, probability = np.asarray(y), np.asarray(probability)
    precision, recall, thresholds = precision_recall_curve(y, probability)
    f1 = (2 * precision * recall / np.maximum(precision + recall, 1e-12))[:-1]
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def evaluate(y, probability, threshold, top_fraction=0.1):
    y, probability = np.asarray(y, dtype=np.int64), np.asarray(probability)
    precision, recall, f1, _ = precision_recall_fscore_support(y, probability >= threshold, average="binary", zero_division=0)
    k = max(1, int(np.ceil(len(y) * top_fraction)))
    top = np.argsort(-probability, kind="stable")[:k]
    return {
        "shops": len(y), "positives": int(y.sum()), "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
        "pr_auc": float(average_precision_score(y, probability)) if y.sum() > 0 else None,
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "threshold": float(threshold), "top_10pct_count": k,
        "precision_at_top_10pct": float(y[top].mean()),
        "recall_at_top_10pct": float(y[top].sum() / max(1, y.sum())),
    }
