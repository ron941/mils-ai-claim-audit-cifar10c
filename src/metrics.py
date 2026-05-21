from __future__ import annotations

import numpy as np


def entropy_from_probs(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def summarize_predictions(
    probs: np.ndarray,
    labels: np.ndarray,
    clean_accuracy: float | None = None,
) -> dict[str, float | int]:
    preds = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    entropy = entropy_from_probs(probs)
    correct = preds == labels
    accuracy = float(correct.mean()) if len(labels) else 0.0
    wrong_confidence = float(confidence[~correct].mean()) if (~correct).any() else 0.0
    clean_acc = accuracy if clean_accuracy is None else float(clean_accuracy)
    accuracy_drop = clean_acc - accuracy
    relative_drop = accuracy_drop / clean_acc if clean_acc > 0 else 0.0
    return {
        "num_samples": int(len(labels)),
        "accuracy": accuracy,
        "clean_accuracy": clean_acc,
        "accuracy_drop": accuracy_drop,
        "relative_drop": relative_drop,
        "avg_confidence": float(confidence.mean()) if len(labels) else 0.0,
        "wrong_confidence": wrong_confidence,
        "prediction_entropy": float(entropy.mean()) if len(labels) else 0.0,
    }


def prediction_arrays(probs: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    preds = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    entropy = entropy_from_probs(probs)
    correct = preds == labels
    return {
        "preds": preds,
        "confidence": confidence,
        "entropy": entropy,
        "correct": correct,
    }

