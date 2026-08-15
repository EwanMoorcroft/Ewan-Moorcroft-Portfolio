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

    class ImageSplit(Dataset):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int):
            row = rows[index]
            path = resolve_image_path(data_root, row["relative_path"])
            with Image.open(path) as source:
                image = source.convert("RGB")
            return transform(image), label_to_index[row["label"]]

    return ImageSplit()
