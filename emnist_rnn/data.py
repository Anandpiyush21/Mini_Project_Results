"""Loading and preprocessing for the EMNIST-Balanced CSV distribution.

The CSV files ship one sample per row: column 0 is the class index, the
remaining 784 columns are the pixels of a 28x28 image. Those pixels are stored
transposed relative to how the characters were written, so every image has to be
mirrored and rotated before it looks like a character.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

IMAGE_SIZE = 28
NUM_CLASSES = 47

TRAIN_CSV = "emnist-balanced-train.csv"
TEST_CSV = "emnist-balanced-test.csv"
MAPPING_TXT = "emnist-balanced-mapping.txt"


@dataclass(frozen=True)
class Split:
    """One preprocessed dataset split."""

    images: np.ndarray  # (n, 28, 28, 1) float32 in [0, 1]
    labels: np.ndarray  # (n, 47) one-hot float32
    classes: np.ndarray  # (n,) integer class indices

    def __len__(self) -> int:
        return len(self.classes)


def deskew(flat_image: np.ndarray) -> np.ndarray:
    """Turn one flat 784-pixel row into an upright 28x28 image."""
    image = flat_image.reshape(IMAGE_SIZE, IMAGE_SIZE)
    return np.rot90(np.fliplr(image))


def load_label_map(data_dir: str) -> dict[int, str]:
    """Map each class index to the character it represents."""
    path = os.path.join(data_dir, MAPPING_TXT)
    mapping = pd.read_csv(path, delimiter=" ", index_col=0, header=None)
    codes = mapping.iloc[:, 0]
    return {index: chr(code) for index, code in enumerate(codes)}


def load_split(data_dir: str, split: str = "train") -> Split:
    """Read a split from disk and apply the full preprocessing pipeline."""
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    path = os.path.join(data_dir, TRAIN_CSV if split == "train" else TEST_CSV)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. See the README for how to download EMNIST-Balanced."
        )

    frame = pd.read_csv(path, header=None)
    classes = frame.iloc[:, 0].to_numpy()
    pixels = frame.iloc[:, 1:].to_numpy()

    images = np.apply_along_axis(deskew, 1, pixels)
    images = (images.astype("float32") / 255.0).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

    one_hot = np.eye(NUM_CLASSES, dtype="float32")[classes]
    return Split(images=images, labels=one_hot, classes=classes)
