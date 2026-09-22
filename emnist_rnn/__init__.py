"""Recurrent architectures for handwritten character recognition on EMNIST-Balanced."""

__version__ = "1.0.0"

from emnist_rnn.data import IMAGE_SIZE, NUM_CLASSES, load_split, load_label_map
from emnist_rnn.models import ARCHITECTURES, build_model

__all__ = [
    "IMAGE_SIZE",
    "NUM_CLASSES",
    "ARCHITECTURES",
    "build_model",
    "load_split",
    "load_label_map",
]
