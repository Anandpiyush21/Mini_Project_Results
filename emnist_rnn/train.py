"""Train one architecture and record its result.

    python -m emnist_rnn.train --arch cnn_bilstm
    python -m emnist_rnn.train --arch all --epochs 30
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import keras
from sklearn.model_selection import train_test_split

from emnist_rnn.data import load_split
from emnist_rnn.models import ARCHITECTURES, build_model

DEFAULT_DATA_DIR = "data/emnist"
DEFAULT_OUT_DIR = "results"
RANDOM_STATE = 88


def train_one(
    architecture: str,
    data_dir: str = DEFAULT_DATA_DIR,
    out_dir: str = DEFAULT_OUT_DIR,
    epochs: int = 30,
    batch_size: int = 32,
    optimizer: str = "adam",
    patience: int = 5,
) -> dict:
    """Train a single architecture and return its metrics."""
    keras.utils.set_random_seed(RANDOM_STATE)

    train = load_split(data_dir, "train")
    test = load_split(data_dir, "test")

    x_train, x_val, y_train, y_val = train_test_split(
        train.images, train.labels, test_size=0.1, random_state=RANDOM_STATE
    )

    model = build_model(architecture)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    model.summary()

    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, f"{architecture}.keras")

    started = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, verbose=1, restore_best_weights=True
            ),
            keras.callbacks.ModelCheckpoint(
                checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1
            ),
        ],
        verbose=1,
    )
    elapsed = time.perf_counter() - started

    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
    test_loss, test_accuracy = model.evaluate(test.images, test.labels, verbose=0)

    result = {
        "architecture": architecture,
        "parameters": model.count_params(),
        "epochs_run": len(history.history["loss"]),
        "train_seconds": round(elapsed, 1),
        "val_accuracy": round(val_accuracy, 4),
        "val_loss": round(val_loss, 4),
        "test_accuracy": round(test_accuracy, 4),
        "test_loss": round(test_loss, 4),
    }

    with open(os.path.join(out_dir, f"{architecture}_history.json"), "w") as handle:
        json.dump(history.history, handle, indent=2)

    print(json.dumps(result, indent=2))
    return result


def append_to_leaderboard(results: list[dict], out_dir: str) -> str:
    """Write/refresh results/results.csv, keeping one row per architecture."""
    path = os.path.join(out_dir, "results.csv")
    rows: dict[str, dict] = {}

    if os.path.exists(path):
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                rows[row["architecture"]] = row

    for result in results:
        rows[result["architecture"]] = {k: str(v) for k, v in result.items()}

    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for architecture in ARCHITECTURES:
            if architecture in rows:
                writer.writerow(rows[architecture])
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--arch", default="cnn_bilstm", choices=(*ARCHITECTURES, "all"),
        help="architecture to train, or 'all' to run the full comparison",
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--optimizer", default="adam", choices=("adam", "adagrad", "adadelta", "rmsprop")
    )
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    targets = ARCHITECTURES if args.arch == "all" else (args.arch,)
    results = [
        train_one(
            architecture,
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            optimizer=args.optimizer,
            patience=args.patience,
        )
        for architecture in targets
    ]

    path = append_to_leaderboard(results, args.out_dir)
    print(f"\nLeaderboard written to {path}")


if __name__ == "__main__":
    main()
