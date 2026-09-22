"""Evaluate a saved model on the EMNIST test split.

    python -m emnist_rnn.evaluate --model results/cnn_bilstm.keras --report
"""

from __future__ import annotations

import argparse

import keras
import numpy as np

from emnist_rnn.data import load_label_map, load_split
from emnist_rnn.train import DEFAULT_DATA_DIR


def evaluate(model_path: str, data_dir: str = DEFAULT_DATA_DIR, report: bool = False) -> dict:
    """Score a checkpoint and optionally print a per-class breakdown."""
    model = keras.models.load_model(model_path)
    test = load_split(data_dir, "test")

    loss, accuracy = model.evaluate(test.images, test.labels, verbose=0)
    print(f"{model_path}: test accuracy {accuracy:.4f}  test loss {loss:.4f}")

    if report:
        from sklearn.metrics import classification_report

        labels = load_label_map(data_dir)
        predicted = np.argmax(model.predict(test.images, verbose=0), axis=1)
        print(
            classification_report(
                test.classes,
                predicted,
                target_names=[labels[i] for i in sorted(labels)],
                digits=3,
                zero_division=0,
            )
        )

    return {"loss": loss, "accuracy": accuracy}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="path to a .keras checkpoint")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--report", action="store_true", help="also print a per-character breakdown"
    )
    args = parser.parse_args()
    evaluate(args.model, args.data_dir, args.report)


if __name__ == "__main__":
    main()
