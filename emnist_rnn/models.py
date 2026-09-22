"""The five architectures compared in this study.

Every model ends with the same classifier head (256 -> 84 -> 47) so that the
only thing that varies between runs is how the 28x28 image is turned into a
feature vector.

Two families are compared:

* **Pure recurrent** (``lstm``, ``bilstm``) read the image directly as a
  sequence of 28 rows, each row a 28-dimensional vector.
* **Convolutional-recurrent** (``cnn_rnn``, ``cnn_lstm``, ``cnn_bilstm``) first
  run two conv/pool blocks, then read the resulting 5x5x48 feature map as a
  sequence of 5 steps.
"""

from __future__ import annotations

from keras import Sequential, layers

from emnist_rnn.data import IMAGE_SIZE, NUM_CLASSES

RECURRENT_UNITS = 64
CONV_FILTERS = (32, 48)
KERNEL_SIZE = (5, 5)

ARCHITECTURES = ("lstm", "bilstm", "cnn_rnn", "cnn_lstm", "cnn_bilstm")


def _classifier_head() -> list[layers.Layer]:
    return [
        layers.Dense(256, activation="relu"),
        layers.Dense(84, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]


def _conv_encoder() -> list[layers.Layer]:
    """Two conv/pool blocks that reduce 28x28x1 to a 5-step sequence."""
    return [
        layers.Conv2D(CONV_FILTERS[0], KERNEL_SIZE, padding="same", activation="relu"),
        layers.MaxPooling2D(strides=2),
        layers.Conv2D(CONV_FILTERS[1], KERNEL_SIZE, padding="valid", activation="relu"),
        layers.MaxPooling2D(strides=2),
        # Each remaining feature-map row becomes one time step.
        layers.TimeDistributed(layers.Flatten()),
    ]


def build_model(architecture: str) -> Sequential:
    """Build one of the five comparison models, uncompiled."""
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"unknown architecture {architecture!r}; expected one of {ARCHITECTURES}"
        )

    body: list[layers.Layer] = [layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))]

    if architecture == "lstm":
        body += [
            layers.Reshape((IMAGE_SIZE, IMAGE_SIZE)),
            layers.LSTM(RECURRENT_UNITS),
        ]
    elif architecture == "bilstm":
        body += [
            layers.Reshape((IMAGE_SIZE, IMAGE_SIZE)),
            layers.Bidirectional(layers.LSTM(RECURRENT_UNITS)),
        ]
    elif architecture == "cnn_rnn":
        body += _conv_encoder() + [
            layers.SimpleRNN(RECURRENT_UNITS, return_sequences=True),
            layers.Flatten(),
        ]
    elif architecture == "cnn_lstm":
        body += _conv_encoder() + [
            layers.LSTM(RECURRENT_UNITS, return_sequences=True),
            layers.Flatten(),
        ]
    else:  # cnn_bilstm
        body += _conv_encoder() + [
            layers.Bidirectional(layers.LSTM(RECURRENT_UNITS, return_sequences=True)),
            layers.Flatten(),
        ]

    return Sequential(body + _classifier_head(), name=architecture)
