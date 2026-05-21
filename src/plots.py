from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate_plots(results: pd.DataFrame, results_tta: pd.DataFrame, overlap: pd.DataFrame, failure_cases: pd.DataFrame, config: dict) -> None:
    figures_dir = Path(config["paths"]["figures_dir"])
    sns.set_theme(style="whitegrid", palette="colorblind")

    corrupted = results[results["condition"] != "clean"].copy()
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.lineplot(data=corrupted, x="severity", y="accuracy", hue="corruption", style="model", marker="o", ax=ax)
    ax.set_title("Accuracy by corruption and severity")
    ax.set_ylim(0, 1)
    _save(fig, figures_dir / "fig1_accuracy_by_corruption.png")

    noise_blur = corrupted[corrupted["corruption"].isin(["gaussian_noise", "motion_blur"])]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=noise_blur, x="severity", y="accuracy_drop", hue="corruption", style="model", marker="o", ax=ax)
    ax.set_title("Accuracy drop: Gaussian noise vs motion blur")
    _save(fig, figures_dir / "fig2_accuracy_drop_noise_vs_blur.png")

    comp = corrupted[corrupted["corruption"].isin(["gaussian_noise", "jpeg_compression"])]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=comp, x="corruption", y="relative_drop", hue="model", errorbar=None, ax=ax)
    ax.set_title("Relative drop comparison")
    _save(fig, figures_dir / "fig3_standard_vs_robust_relative_drop.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=corrupted, x="corruption", y="wrong_confidence", hue="model", errorbar=None, ax=ax)
    ax.set_title("Mean confidence on wrong predictions")
    ax.tick_params(axis="x", rotation=25)
    _save(fig, figures_dir / "fig4_wrong_confidence.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=overlap, x="severity", y="overlap_error_union", hue="corruption", marker="o", ax=ax)
    ax.set_title("Failure overlap by severity")
    ax.set_ylim(0, 1)
    _save(fig, figures_dir / "fig5_failure_overlap_by_severity.png")

    base_worst = corrupted.groupby("model", as_index=False)["accuracy"].min().assign(setting="baseline")
    tta_worst = results_tta[results_tta["condition"] != "clean"].groupby("model", as_index=False)["accuracy"].min().assign(setting="tta")
    worst = pd.concat([base_worst, tta_worst], ignore_index=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=worst, x="model", y="accuracy", hue="setting", errorbar=None, ax=ax)
    ax.set_title("Worst-condition accuracy before/after TTA")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=15)
    _save(fig, figures_dir / "fig6_tta_before_after.png")

