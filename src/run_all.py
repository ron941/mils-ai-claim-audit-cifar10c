from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from .claim_audit import generate_claim_audit
from .datasets import build_conditions, ensure_cifar10, ensure_cifar10c, seed_everything, validate_cifar10c_slices
from .evaluate import evaluate_models, run_with_batch_retry
from .failure_cases import export_failure_cases
from .failure_overlap import compute_failure_overlap
from .models import load_model_pair
from .plots import generate_plots


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_paths(config: dict, root: Path) -> dict:
    resolved = dict(config)
    resolved["paths"] = dict(config["paths"])
    for key, value in config["paths"].items():
        path = Path(value)
        resolved["paths"][key] = str(path if path.is_absolute() else root / path)
    return resolved


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_output_dirs(config: dict) -> None:
    for key in ["output_dir", "predictions_dir", "figures_dir", "failure_cases_dir"]:
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)


def choose_device(config: dict) -> torch.device:
    requested = str(config["runtime"].get("device", "cuda:0"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU")
        return torch.device("cpu")
    return torch.device(requested)


def validate_outputs(config: dict) -> dict:
    output_dir = Path(config["paths"]["output_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    checks = {
        "results_rows": len(pd.read_csv(output_dir / "results.csv")),
        "results_tta_rows": len(pd.read_csv(output_dir / "results_tta.csv")),
        "failure_overlap_rows": len(pd.read_csv(output_dir / "failure_overlap.csv")),
        "claim_audit_rows": len(pd.read_csv(output_dir / "claim_audit.csv")),
        "failure_cases_rows": len(pd.read_csv(output_dir / "failure_cases.csv")),
        "figures": {},
    }
    for name in [
        "fig1_accuracy_by_corruption.png",
        "fig2_accuracy_drop_noise_vs_blur.png",
        "fig3_standard_vs_robust_relative_drop.png",
        "fig4_wrong_confidence.png",
        "fig5_failure_overlap_by_severity.png",
        "fig6_tta_before_after.png",
    ]:
        path = figures_dir / name
        checks["figures"][name] = path.exists() and path.stat().st_size > 0
    (output_dir / "validation_summary.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CIFAR-10-C robustness audit")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument("--max-samples", type=int, default=None, help="Override sample count per condition")
    parser.add_argument("--skip-tta", action="store_true", help="Skip TTA evaluation")
    args = parser.parse_args()

    root = _project_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = _resolve_paths(load_config(config_path), root)
    if args.max_samples is not None:
        config["data"]["max_samples"] = args.max_samples
    ensure_output_dirs(config)
    seed_everything(int(config["project"].get("seed", 42)))

    cifar10_dir = Path(config["paths"]["cifar10_dir"])
    cifar10c_dir = Path(config["paths"]["cifar10c_dir"])
    ensure_cifar10(cifar10_dir)
    cifar10c_root = ensure_cifar10c(cifar10c_dir, config["data"]["cifar10c_url"])
    config["paths"]["cifar10c_dir"] = str(cifar10c_root)
    validate_cifar10c_slices(cifar10c_root, config["data"]["corruptions"], config["data"]["severities"])

    conditions = build_conditions(config["data"]["corruptions"], config["data"]["severities"])
    device = choose_device(config)
    models, model_metadata = load_model_pair(config, device, Path(config["paths"]["output_dir"]))

    results = run_with_batch_retry(lambda: evaluate_models(models, conditions, config, device, use_tta=False), config)
    output_dir = Path(config["paths"]["output_dir"])
    results.to_csv(output_dir / "results.csv", index=False)

    if args.skip_tta:
        results_tta = results.copy()
        results_tta["tta"] = False
    else:
        results_tta = run_with_batch_retry(lambda: evaluate_models(models, conditions, config, device, use_tta=True), config)
    results_tta.to_csv(output_dir / "results_tta.csv", index=False)

    overlap = compute_failure_overlap(models, conditions, config)
    overlap.to_csv(output_dir / "failure_overlap.csv", index=False)

    failure_cases = export_failure_cases(models, conditions, config)
    failure_cases.to_csv(output_dir / "failure_cases.csv", index=False)

    generate_plots(results, results_tta, overlap, failure_cases, config)

    claim_audit = generate_claim_audit(results, results_tta, overlap, failure_cases, model_metadata, config)
    claim_audit.to_csv(output_dir / "claim_audit.csv", index=False)

    checks = validate_outputs(config)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()

