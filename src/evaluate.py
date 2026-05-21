from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import Condition, get_dataset
from .metrics import prediction_arrays, summarize_predictions
from .models import LoadedModel, model_slug
from .tta import tta_probabilities


def logits_to_probs(output: torch.Tensor) -> torch.Tensor:
    row_sums = output.sum(dim=1)
    if bool((output >= 0).all()) and torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3):
        return output
    return torch.softmax(output, dim=1)


def evaluate_condition(
    loaded_model: LoadedModel,
    condition: Condition,
    config: dict,
    device: torch.device,
    clean_accuracy: float | None,
    use_tta: bool = False,
) -> tuple[dict, dict[str, np.ndarray]]:
    paths = config["paths"]
    dataset = get_dataset(
        condition,
        Path(paths["cifar10_dir"]),
        Path(paths["cifar10c_dir"]),
        config["data"].get("max_samples"),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["runtime"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["runtime"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    probs_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    indices_parts: list[np.ndarray] = []
    loaded_model.model.eval()
    with torch.inference_mode():
        for images, labels, indices in tqdm(loader, desc=f"{loaded_model.name}:{condition.condition}:tta={use_tta}"):
            images = images.to(device, non_blocking=True)
            if use_tta:
                probs = tta_probabilities(
                    loaded_model.model,
                    images,
                    horizontal_flip=bool(config["tta"].get("horizontal_flip", True)),
                )
            else:
                probs = logits_to_probs(loaded_model.model(images))
            probs_parts.append(probs.detach().cpu().numpy())
            labels_parts.append(labels.numpy())
            indices_parts.append(indices.numpy())

    probs_np = np.concatenate(probs_parts, axis=0)
    labels_np = np.concatenate(labels_parts, axis=0).astype(np.int64)
    indices_np = np.concatenate(indices_parts, axis=0).astype(np.int64)
    summary = summarize_predictions(probs_np, labels_np, clean_accuracy=clean_accuracy)
    row = {
        "model": loaded_model.name,
        "condition": condition.condition,
        "corruption": condition.corruption,
        "severity": condition.severity if condition.severity is not None else 0,
        **summary,
    }
    arrays = {"probs": probs_np, "labels": labels_np, "indices": indices_np, **prediction_arrays(probs_np, labels_np)}
    return row, arrays


def save_predictions(
    arrays: dict[str, np.ndarray],
    loaded_model: LoadedModel,
    condition: Condition,
    config: dict,
    use_tta: bool = False,
) -> Path:
    predictions_dir = Path(config["paths"]["predictions_dir"])
    predictions_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_tta" if use_tta else ""
    path = predictions_dir / f"{model_slug(loaded_model)}__{condition.condition}{suffix}.npz"
    np.savez_compressed(path, **arrays)
    return path


def evaluate_models(
    models: list[LoadedModel],
    conditions: list[Condition],
    config: dict,
    device: torch.device,
    use_tta: bool = False,
) -> pd.DataFrame:
    rows: list[dict] = []
    clean_accuracies: dict[str, float] = {}

    clean_condition = conditions[0]
    for loaded_model in models:
        row, arrays = evaluate_condition(loaded_model, clean_condition, config, device, clean_accuracy=None, use_tta=use_tta)
        clean_accuracies[loaded_model.name] = float(row["accuracy"])
        save_predictions(arrays, loaded_model, clean_condition, config, use_tta=use_tta)
        rows.append(row)

    for condition in conditions[1:]:
        for loaded_model in models:
            row, arrays = evaluate_condition(
                loaded_model,
                condition,
                config,
                device,
                clean_accuracy=clean_accuracies[loaded_model.name],
                use_tta=use_tta,
            )
            save_predictions(arrays, loaded_model, condition, config, use_tta=use_tta)
            rows.append(row)

    df = pd.DataFrame(rows)
    if use_tta:
        df["tta"] = True
    return df


def run_with_batch_retry(fn: Callable[[], pd.DataFrame], config: dict) -> pd.DataFrame:
    try:
        return fn()
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or int(config["runtime"]["batch_size"]) <= 256:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("CUDA OOM at batch_size", config["runtime"]["batch_size"], "retrying with 256")
        config["runtime"]["batch_size"] = 256
        return fn()

