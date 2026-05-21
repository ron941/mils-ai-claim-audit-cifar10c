from __future__ import annotations

import os
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from tqdm import tqdm


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


@dataclass(frozen=True)
class Condition:
    condition: str
    corruption: str
    severity: int | None


class DownloadProgress(tqdm):
    def update_to(self, blocks: int = 1, block_size: int = 1, total_size: int | None = None) -> None:
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def ensure_cifar10(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    datasets.CIFAR10(root=str(root), train=False, download=True)


def _find_cifar10c_root(path: Path) -> Path | None:
    candidates = [path, path / "CIFAR-10-C", path / "cifar-10-c"]
    for candidate in candidates:
        if (candidate / "labels.npy").exists() and any(candidate.glob("*.npy")):
            return candidate
    for labels_path in path.rglob("labels.npy"):
        candidate = labels_path.parent
        if any(candidate.glob("*.npy")):
            return candidate
    return None


def ensure_cifar10c(root: Path, url: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing = _find_cifar10c_root(root)
    if existing is not None:
        return existing

    archive_path = root / "CIFAR-10-C.tar"
    partial_path = root / "CIFAR-10-C.tar.part"
    if not archive_path.exists():
        print(f"Downloading CIFAR-10-C to {archive_path}")
        with DownloadProgress(unit="B", unit_scale=True, miniters=1, desc="CIFAR-10-C") as progress:
            urllib.request.urlretrieve(url, filename=partial_path, reporthook=progress.update_to)
        partial_path.replace(archive_path)

    print(f"Extracting {archive_path}")
    try:
        with tarfile.open(archive_path) as tar:
            tar.extractall(root)
    except tarfile.TarError:
        archive_path.unlink(missing_ok=True)
        raise

    extracted = _find_cifar10c_root(root)
    if extracted is None:
        raise FileNotFoundError(f"Could not find extracted CIFAR-10-C files under {root}")
    return extracted


def build_conditions(corruptions: Iterable[str], severities: Iterable[int]) -> list[Condition]:
    conditions = [Condition("clean", "clean", None)]
    for corruption in corruptions:
        for severity in severities:
            conditions.append(Condition(f"{corruption}_s{severity}", corruption, int(severity)))
    return conditions


class CleanCIFAR10Dataset(Dataset):
    def __init__(self, root: Path, max_samples: int | None = None) -> None:
        self.dataset = datasets.CIFAR10(
            root=str(root),
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        self.max_samples = min(max_samples, len(self.dataset)) if max_samples else len(self.dataset)

    def __len__(self) -> int:
        return self.max_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        image, label = self.dataset[index]
        return image, int(label), int(index)

    def image_for_index(self, index: int) -> Image.Image:
        image, _ = self.dataset[index]
        return transforms.ToPILImage()(image)


class CIFAR10CDataset(Dataset):
    def __init__(self, root: Path, corruption: str, severity: int, max_samples: int | None = None) -> None:
        self.root = root
        self.corruption = corruption
        self.severity = int(severity)
        corruption_path = root / f"{corruption}.npy"
        labels_path = root / "labels.npy"
        if not corruption_path.exists():
            raise FileNotFoundError(f"Missing CIFAR-10-C corruption file: {corruption_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing CIFAR-10-C labels file: {labels_path}")

        start = (self.severity - 1) * 10000
        end = self.severity * 10000
        images = np.load(corruption_path, mmap_mode="r")
        labels = np.load(labels_path)

        if images.shape[0] < end:
            raise ValueError(f"{corruption_path} has {images.shape[0]} images; severity {severity} needs {end}")

        self.images = images[start:end]
        if len(labels) == images.shape[0]:
            self.labels = labels[start:end]
        elif len(labels) == 10000:
            self.labels = labels
        else:
            raise ValueError(f"Unexpected labels length {len(labels)} for CIFAR-10-C")

        self.max_samples = min(max_samples, len(self.images)) if max_samples else len(self.images)

    def __len__(self) -> int:
        return self.max_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        image = Image.fromarray(np.asarray(self.images[index]).astype(np.uint8))
        tensor = transforms.ToTensor()(image)
        return tensor, int(self.labels[index]), int(index)

    def image_for_index(self, index: int) -> Image.Image:
        return Image.fromarray(np.asarray(self.images[index]).astype(np.uint8))


def get_dataset(
    condition: Condition,
    cifar10_dir: Path,
    cifar10c_dir: Path,
    max_samples: int | None,
) -> CleanCIFAR10Dataset | CIFAR10CDataset:
    if condition.condition == "clean":
        return CleanCIFAR10Dataset(cifar10_dir, max_samples=max_samples)
    if condition.severity is None:
        raise ValueError("Corrupted condition requires a severity")
    return CIFAR10CDataset(cifar10c_dir, condition.corruption, condition.severity, max_samples=max_samples)


def validate_cifar10c_slices(root: Path, corruptions: Iterable[str], severities: Iterable[int]) -> None:
    labels = np.load(root / "labels.npy")
    for corruption in corruptions:
        images = np.load(root / f"{corruption}.npy", mmap_mode="r")
        for severity in severities:
            start = (int(severity) - 1) * 10000
            end = int(severity) * 10000
            if images[start:end].shape[0] != 10000:
                raise ValueError(f"Bad slice for {corruption} severity {severity}")
    if len(labels) not in {10000, 50000}:
        raise ValueError(f"Expected CIFAR-10-C labels length 10000 or 50000, got {len(labels)}")


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
