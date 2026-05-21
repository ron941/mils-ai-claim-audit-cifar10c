# CIFAR-10-C Robustness Audit

This repository contains a MILS AI-Claim Audit project for robustness under CIFAR-10-C distribution shifts. It compares a standard RobustBench model with a robust AugMix-WRN model, audits five AI-generated claims, and links each conclusion to CSV results, figures, or failure cases.

Repository: https://github.com/ron941/mils-ai-claim-audit-cifar10c

## Project Summary

- Dataset: CIFAR-10 test set and CIFAR-10-C.
- Models: RobustBench `Standard` and `Hendrycks2020AugMix_WRN`.
- Corruptions: Gaussian noise, motion blur, fog, contrast, and JPEG compression.
- Severities: 1, 3, and 5.
- Claims audited: C1-C5, covering noise vs blur, JPEG robustness, high-confidence wrong predictions, TTA, and failure overlap.

## What The Code Produces

- `outputs/results.csv`
- `outputs/results_tta.csv`
- `outputs/failure_overlap.csv`
- `outputs/failure_cases.csv`
- `outputs/claim_audit.csv`
- `outputs/figures/*.png`
- `outputs/failure_cases/*.png`
- `notebooks/MILS_AI_Claim_Audit.ipynb`
- `report/MILS_AI_Claim_Audit_Report_Word.docx`

## Repository Notes

The repository keeps source code, configs, the final report, compact CSV outputs, figures, representative failure-case images, and report assets so the evidence is easy to inspect on GitHub. Large downloaded datasets, RobustBench model weights, and per-sample prediction caches are excluded by `.gitignore`.

For course upload, use the compact `submission_minimal/` folder generated in this workspace. It contains the final report and necessary code only, following the instruction not to upload code execution artifacts or datasets.

## Repository Layout

```text
configs/           Experiment configuration
src/               Data loading, model loading, evaluation, plotting, and audit code
notebooks/         Notebook view of the generated evidence
outputs/           Compact CSV outputs, figures, and representative failure cases
report/            Final report document
report_assets/     Figure montages inserted into the report
```

## Setup

```bash
cd /raid/ron/HW/robustness_audit
python3 -m pip install -r requirements.txt
```

RobustBench imports the `autoattack` module at import time, so the requirements install AutoAttack from its upstream GitHub repository.

The pipeline downloads CIFAR-10 through torchvision and CIFAR-10-C from a Hugging Face mirror of the original CIFAR-10-C tarball:

```text
https://huggingface.co/datasets/torch-uncertainty/CIFAR-C/resolve/main/CIFAR-10-C.tar
```

The original CIFAR-10-C release is hosted on Zenodo at `https://zenodo.org/records/2535967`; the mirror is used because it is much faster on this machine.

Primary models are loaded from RobustBench:

- `Standard`, CIFAR-10 corruptions threat model
- `Hendrycks2020AugMix_WRN`, CIFAR-10 corruptions threat model

If RobustBench or its weights fail, the pipeline falls back to CIFAR-10 pretrained models from `chenyaofo/pytorch-cifar-models`. In that fallback case, C2 is worded as a model-comparison robustness claim rather than as a robust-model claim.

## Run

Smoke test:

```bash
python3 -m src.run_all --config configs/config.yaml --max-samples 64
```

Full evidence run:

```bash
python3 -m src.run_all --config configs/config.yaml
```

The default device is `cuda:0` and default batch size is `512`. If CUDA runs out of memory, the runner retries with batch size `256`.

## Claims

- C1: Gaussian noise hurts CNNs more than motion blur.
- C2: Robust/AugMix model is more stable under noise, but may not be better under JPEG compression.
- C3: Corrupted images can produce high-confidence wrong predictions.
- C4: TTA may improve worst-condition accuracy but hurt clean accuracy or some conditions.
- C5: Failure overlap between models increases as severity increases.

## Metrics

- Accuracy
- Accuracy drop
- Relative drop
- Average confidence
- Wrong confidence
- Prediction entropy
- Worst-condition accuracy
- Failure overlap over all samples
- Failure overlap over the union of failures
