from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import Condition
from .models import LoadedModel, model_slug


def compute_failure_overlap(models: list[LoadedModel], conditions: list[Condition], config: dict) -> pd.DataFrame:
    if len(models) != 2:
        raise ValueError("Failure overlap requires exactly two models")
    predictions_dir = Path(config["paths"]["predictions_dir"])
    rows: list[dict] = []
    for condition in conditions:
        if condition.condition == "clean":
            continue
        first = np.load(predictions_dir / f"{model_slug(models[0])}__{condition.condition}.npz")
        second = np.load(predictions_dir / f"{model_slug(models[1])}__{condition.condition}.npz")
        first_wrong = ~first["correct"].astype(bool)
        second_wrong = ~second["correct"].astype(bool)
        both_wrong = first_wrong & second_wrong
        either_wrong = first_wrong | second_wrong
        total = len(first_wrong)
        rows.append(
            {
                "condition": condition.condition,
                "corruption": condition.corruption,
                "severity": condition.severity,
                "model_a": models[0].name,
                "model_b": models[1].name,
                "both_wrong": int(both_wrong.sum()),
                "either_wrong": int(either_wrong.sum()),
                "total": int(total),
                "overlap_all": float(both_wrong.sum() / total) if total else 0.0,
                "overlap_error_union": float(both_wrong.sum() / either_wrong.sum()) if either_wrong.any() else 0.0,
            }
        )
    return pd.DataFrame(rows)

