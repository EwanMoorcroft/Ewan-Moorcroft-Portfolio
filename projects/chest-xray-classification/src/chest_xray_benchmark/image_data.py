"""Split-backed image loading and transforms."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

LABEL_ORDER = ("Normal", "Lung Opacity", "Viral Pneumonia")


def resolve_image_path(data_root: Path, relative_path: str) -> Path:
    root = data_root.resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Split file contains a path outside the data root")
    return path


def verify_split_image_identity(rows: list[dict[str, str]], data_root: Path) -> list[Path]:
    """Resolve and hash every split image before a training or evaluation loader is built."""

    from .manifest import sha256_file

    verified: list[Path] = []
    for index, row in enumerate(rows, start=2):
        relative_path = row["relative_path"]
        path = resolve_image_path(data_root, relative_path)
        if not path.is_file():
            raise ValueError(f"Split row {index} image does not exist: {relative_path}")
        try:
            expected_size = int(row["byte_size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Split row {index} has an invalid byte size") from exc
        if expected_size < 0 or path.stat().st_size != expected_size:
            raise ValueError(f"Split row {index} image size differs from the manifest")
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"Split row {index} image SHA-256 differs from the manifest")
        verified.append(path)
    return verified


def build_transforms(image_size: int, rotation_degrees: float):
    from torchvision import transforms

    normalization = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomRotation(rotation_degrees),
            transforms.ToTensor(),
            normalization,
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return train_transform, evaluation_transform


def build_image_dataset(
    rows: list[dict[str, str]],
    data_root: Path,
    transform: Callable,
):
    from PIL import Image
    from torch.utils.data import Dataset

    label_to_index = {label: index for index, label in enumerate(LABEL_ORDER)}
    unknown = sorted({row["label"] for row in rows} - set(label_to_index))
    if unknown:
        raise ValueError(f"Unknown labels: {', '.join(unknown)}")

    verified_paths = verify_split_image_identity(rows, data_root)

    class ImageSplit(Dataset):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int):
            row = rows[index]
            path = verified_paths[index]
            with Image.open(path) as source:
                image = source.convert("RGB")
            return transform(image), label_to_index[row["label"]]

    return ImageSplit()
