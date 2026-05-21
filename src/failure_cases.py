from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .datasets import CIFAR10_CLASSES, Condition, get_dataset
from .models import LoadedModel, model_slug


def _condition_lookup(conditions: list[Condition]) -> dict[str, Condition]:
    return {condition.condition: condition for condition in conditions}


def _draw_failure_image(
    image: Image.Image,
    path: Path,
    model: str,
    condition: str,
    true_label: str,
    pred_label: str,
    confidence: float,
    entropy: float,
) -> None:
    scale = 8
    image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST).convert("RGB")
    header_h = 72
    canvas = Image.new("RGB", (image.width, image.height + header_h), "white")
    canvas.paste(image, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    lines = [
        f"{model} | {condition}",
        f"true={true_label} pred={pred_label} conf={confidence:.3f} entropy={entropy:.3f}",
    ]
    y = 8
    for line in lines:
        draw.text((8, y), line, fill=(0, 0, 0))
        y += 26
    canvas.save(path)


def export_failure_cases(models: list[LoadedModel], conditions: list[Condition], config: dict) -> pd.DataFrame:
    predictions_dir = Path(config["paths"]["predictions_dir"])
    output_dir = Path(config["paths"]["failure_cases_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    per_model = int(config["failure_cases"]["per_model"])
    threshold = float(config["failure_cases"]["high_confidence_threshold"])
    lookup = _condition_lookup(conditions)
    rows: list[dict] = []

    for loaded_model in models:
        candidates: list[dict] = []
        for condition in conditions:
            if condition.condition == "clean":
                continue
            data = np.load(predictions_dir / f"{model_slug(loaded_model)}__{condition.condition}.npz")
            wrong_positions = np.where(~data["correct"].astype(bool))[0]
            for pos in wrong_positions:
                candidates.append(
                    {
                        "model": loaded_model.name,
                        "condition": condition.condition,
                        "corruption": condition.corruption,
                        "severity": condition.severity,
                        "index": int(data["indices"][pos]),
                        "true_label": int(data["labels"][pos]),
                        "pred_label": int(data["preds"][pos]),
                        "confidence": float(data["confidence"][pos]),
                        "entropy": float(data["entropy"][pos]),
                        "above_threshold": bool(float(data["confidence"][pos]) >= threshold),
                    }
                )
        candidates.sort(key=lambda item: item["confidence"], reverse=True)
        selected = [item for item in candidates if item["above_threshold"]][:per_model]
        if len(selected) < per_model:
            selected = (selected + [item for item in candidates if not item["above_threshold"]])[:per_model]

        for rank, item in enumerate(selected, start=1):
            condition = lookup[item["condition"]]
            dataset = get_dataset(
                condition,
                Path(config["paths"]["cifar10_dir"]),
                Path(config["paths"]["cifar10c_dir"]),
                config["data"].get("max_samples"),
            )
            image = dataset.image_for_index(item["index"])
            image_path = output_dir / f"{model_slug(loaded_model)}__failure_{rank:02d}__{item['condition']}__idx{item['index']}.png"
            _draw_failure_image(
                image,
                image_path,
                item["model"],
                item["condition"],
                CIFAR10_CLASSES[item["true_label"]],
                CIFAR10_CLASSES[item["pred_label"]],
                item["confidence"],
                item["entropy"],
            )
            rows.append({"image_path": str(image_path), **item})

    return pd.DataFrame(rows)

