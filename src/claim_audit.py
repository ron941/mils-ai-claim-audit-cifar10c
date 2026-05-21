from __future__ import annotations

from pathlib import Path

import pandas as pd


def _decision(supported: bool, partial: bool = False) -> str:
    if supported:
        return "Supported"
    if partial:
        return "Partially supported"
    return "Refuted"


def generate_claim_audit(
    results: pd.DataFrame,
    results_tta: pd.DataFrame,
    overlap: pd.DataFrame,
    failure_cases: pd.DataFrame,
    model_metadata: dict,
    config: dict,
) -> pd.DataFrame:
    figures_dir = Path(config["paths"]["figures_dir"])
    used_fallback = bool(model_metadata.get("used_fallback", False))
    model_names = list(results["model"].drop_duplicates())
    standard_model = model_names[0]
    comparison_model = model_names[1]
    corrupted = results[results["condition"] != "clean"]

    std = corrupted[corrupted["model"] == standard_model]
    noise = std[std["corruption"] == "gaussian_noise"].set_index("severity")
    blur = std[std["corruption"] == "motion_blur"].set_index("severity")
    comparisons = [
        float(noise.loc[sev, "accuracy_drop"]) > float(blur.loc[sev, "accuracy_drop"])
        for sev in sorted(set(noise.index) & set(blur.index))
    ]
    c1_supported = sum(comparisons) >= 2 and noise["accuracy_drop"].mean() > blur["accuracy_drop"].mean()
    c1_partial = any(comparisons)

    comp = corrupted[corrupted["corruption"].isin(["gaussian_noise", "jpeg_compression"])]
    rel = comp.groupby(["model", "corruption"])["relative_drop"].mean().unstack()
    gaussian_adv = rel.loc[comparison_model, "gaussian_noise"] < rel.loc[standard_model, "gaussian_noise"]
    jpeg_adv = rel.loc[comparison_model, "jpeg_compression"] < rel.loc[standard_model, "jpeg_compression"]
    c2_supported = bool(gaussian_adv and (not jpeg_adv or rel.loc[comparison_model, "jpeg_compression"] >= rel.loc[comparison_model, "gaussian_noise"]))
    c2_partial = bool(gaussian_adv or not jpeg_adv)

    c3_pool = failure_cases[failure_cases["corruption"].isin(["fog", "contrast", "gaussian_noise"])]
    c3_supported = bool((c3_pool["confidence"] > 0.8).any()) if not c3_pool.empty else False

    base_clean = results[results["condition"] == "clean"].set_index("model")["accuracy"]
    tta_clean = results_tta[results_tta["condition"] == "clean"].set_index("model")["accuracy"]
    base_worst = corrupted.groupby("model")["accuracy"].min()
    tta_worst = results_tta[results_tta["condition"] != "clean"].groupby("model")["accuracy"].min()
    worst_improved = bool((tta_worst > base_worst).any())
    clean_hurt = bool((tta_clean < base_clean).any())
    merged_acc = results.merge(results_tta, on=["model", "condition"], suffixes=("_base", "_tta"))
    some_condition_hurt = bool((merged_acc["accuracy_tta"] < merged_acc["accuracy_base"]).any())
    c4_supported = worst_improved and (clean_hurt or some_condition_hurt)
    c4_partial = worst_improved or clean_hurt or some_condition_hurt

    mean_overlap = overlap.groupby("severity")["overlap_error_union"].mean()
    severity_increase = mean_overlap.get(5, 0.0) > mean_overlap.get(1, 0.0)
    by_corruption = overlap.pivot(index="corruption", columns="severity", values="overlap_error_union")
    increases = int((by_corruption.get(5, 0.0) > by_corruption.get(1, 0.0)).sum())
    c5_supported = bool(severity_increase and increases >= 3)
    c5_partial = bool(severity_increase or increases >= 2)

    c2_claim = (
        "A model that is more stable under Gaussian noise may not be better under JPEG compression."
        if used_fallback
        else "Robust/AugMix model is more stable under noise, but may not be better under JPEG compression."
    )
    c2_hypothesis = (
        f"{comparison_model} has lower Gaussian relative drop than {standard_model}, while JPEG advantage is absent or weaker."
    )

    rows = [
        {
            "claim_id": "C1",
            "ai_claim": "Gaussian noise hurts CNNs more than motion blur.",
            "testable_hypothesis": "For the standard model, Gaussian-noise accuracy drop exceeds motion-blur drop at most severities and on average.",
            "evidence_file": str(figures_dir / "fig2_accuracy_drop_noise_vs_blur.png"),
            "audit_decision": _decision(c1_supported, c1_partial),
            "summary": f"Gaussian drop exceeded motion blur in {sum(comparisons)}/{len(comparisons)} severities.",
        },
        {
            "claim_id": "C2",
            "ai_claim": c2_claim,
            "testable_hypothesis": c2_hypothesis,
            "evidence_file": str(figures_dir / "fig3_standard_vs_robust_relative_drop.png"),
            "audit_decision": _decision(c2_supported, c2_partial),
            "summary": f"Mean relative drops: {rel.to_dict()}",
        },
        {
            "claim_id": "C3",
            "ai_claim": "Corrupted images can produce high-confidence wrong predictions.",
            "testable_hypothesis": "Fog, contrast, or Gaussian-noise wrong examples include confidence > 0.8.",
            "evidence_file": str(Path(config["paths"]["output_dir"]) / "failure_cases.csv"),
            "audit_decision": _decision(c3_supported),
            "summary": f"Max selected failure confidence: {failure_cases['confidence'].max() if not failure_cases.empty else 0:.3f}.",
        },
        {
            "claim_id": "C4",
            "ai_claim": "TTA may improve worst-condition accuracy but hurt clean accuracy or some conditions.",
            "testable_hypothesis": "After TTA, worst-condition accuracy improves and clean or another condition decreases.",
            "evidence_file": str(figures_dir / "fig6_tta_before_after.png"),
            "audit_decision": _decision(c4_supported, c4_partial),
            "summary": f"worst_improved={worst_improved}, clean_hurt={clean_hurt}, any_condition_hurt={some_condition_hurt}.",
        },
        {
            "claim_id": "C5",
            "ai_claim": "Failure overlap between models increases as severity increases.",
            "testable_hypothesis": "Mean severity-5 overlap exceeds severity-1 overlap and at least 3/5 corruptions increase.",
            "evidence_file": str(figures_dir / "fig5_failure_overlap_by_severity.png"),
            "audit_decision": _decision(c5_supported, c5_partial),
            "summary": f"Mean severity overlap: {mean_overlap.to_dict()}; increasing corruptions={increases}/5.",
        },
    ]
    return pd.DataFrame(rows)

