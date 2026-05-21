from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass
class LoadedModel:
    name: str
    model: torch.nn.Module
    source: str
    is_robust: bool


class CIFARNormalize(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class NormalizedModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.normalize = CIFARNormalize()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.normalize(x))


def _safe_name(name: str) -> str:
    return name.lower().replace("/", "_").replace(" ", "_")


def _load_robustbench_pair(config: dict, device: torch.device) -> list[LoadedModel]:
    from robustbench.utils import load_model

    rb_config = config["models"]["robustbench"]
    standard_id = rb_config["standard"]
    robust_id = rb_config["robust"]
    standard = load_model(standard_id, dataset="cifar10", threat_model="corruptions").to(device).eval()
    robust = load_model(robust_id, dataset="cifar10", threat_model="corruptions").to(device).eval()
    return [
        LoadedModel("standard", standard, f"robustbench:{standard_id}", is_robust=False),
        LoadedModel("robust_augmix_wrn", robust, f"robustbench:{robust_id}", is_robust=True),
    ]


def _load_torchhub_model(model_name: str, device: torch.device) -> torch.nn.Module:
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        model_name,
        pretrained=True,
        trust_repo=True,
    )
    return NormalizedModel(model).to(device).eval()


def _load_fallback_pair(config: dict, device: torch.device) -> list[LoadedModel]:
    fallback = config["models"]["fallback"]
    standard_id = fallback["standard"]
    alternate_id = fallback["alternate"]
    standard = _load_torchhub_model(standard_id, device)
    alternate = _load_torchhub_model(alternate_id, device)
    return [
        LoadedModel("standard_resnet56", standard, f"torchhub:chenyaofo/{standard_id}", is_robust=False),
        LoadedModel("fallback_vgg16_bn", alternate, f"torchhub:chenyaofo/{alternate_id}", is_robust=False),
    ]


def load_model_pair(config: dict, device: torch.device, output_dir: Path) -> tuple[list[LoadedModel], dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict = {"used_fallback": False, "fallback_reason": "", "models": []}
    if config["models"].get("prefer_robustbench", True):
        try:
            models = _load_robustbench_pair(config, device)
            metadata["models"] = [
                {"name": m.name, "source": m.source, "is_robust": m.is_robust} for m in models
            ]
            (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            return models, metadata
        except Exception as exc:  # noqa: BLE001 - fallback is intentional for assignment robustness.
            metadata["used_fallback"] = True
            metadata["fallback_reason"] = repr(exc)

    models = _load_fallback_pair(config, device)
    metadata["models"] = [{"name": m.name, "source": m.source, "is_robust": m.is_robust} for m in models]
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return models, metadata


def model_slug(model: LoadedModel | str) -> str:
    if isinstance(model, LoadedModel):
        return _safe_name(model.name)
    return _safe_name(model)

