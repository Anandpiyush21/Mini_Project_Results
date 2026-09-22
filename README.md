# Reading Handwriting as a Sequence

**A controlled comparison of five recurrent architectures on EMNIST-Balanced (47 classes).**

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.16%2B-orange">
  <img alt="Keras" src="https://img.shields.io/badge/Keras-3-red">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

Convolutional networks are the default answer for handwritten character
recognition. This project asks a narrower question: **how much of that job can a
recurrent network do on its own, and what does adding convolution actually buy
you?**

A 28×28 character image can be read as a sequence — 28 rows, one per time step —
which makes it a legitimate input for an RNN. Five models are trained under
identical conditions, differing *only* in how the image becomes a feature
vector. Everything downstream (the 256 → 84 → 47 classifier head, the optimiser,
the early-stopping rule, the split seed) is held fixed, so the accuracy gaps are
attributable to the encoder alone.

---

## Results

Trained on EMNIST-Balanced with Adam, batch size 32, early stopping on
validation loss (patience 5). Accuracy is on the held-out 18,800-image test set.

| Architecture   | Encoder                          | Params  | Epochs | Val acc | **Test acc** | Test loss |
| :------------- | :------------------------------- | ------: | -----: | ------: | -----------: | --------: |
| `lstm`         | LSTM(64) over 28 rows            |  66,031 |     19 |  0.8618 |   **0.8643** |    0.4060 |
| `bilstm`       | BiLSTM(64) over 28 rows          | 106,223 |     16 |  0.8598 |   **0.8635** |    0.4009 |
| `cnn_rnn`      | 2× conv/pool → SimpleRNN(64)     | 166,559 |     10 |  0.8708 |   **0.8693** |    0.3879 |
| `cnn_lstm`     | 2× conv/pool → LSTM(64)          | 225,119 |      9 |  0.8752 |   **0.8748** |    0.3593 |
| `cnn_bilstm`   | 2× conv/pool → BiLSTM(64)        | 385,119 |      9 |  0.8761 | **0.8762** 🏆 |    0.3599 |

```
lstm        ███████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░  86.43%
bilstm      ██████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░  86.35%
cnn_rnn     ███████████████████████████████░░░░░░░░░░░░░░░░░  86.93%
cnn_lstm    ████████████████████████████████████████░░░░░░░░  87.48%
cnn_bilstm  ██████████████████████████████████████████░░░░░░  87.62%
            └──────────────────────────────────────────────┘
            85%                                        88%
```

### What the numbers say

**Convolution helps, but less than you would expect.** The best hybrid beats the
plain LSTM by 1.2 points while carrying roughly six times the parameters. Read
row by row, an LSTM alone already recovers most of the signal — the recurrent
encoder is doing real work, not just consuming a convnet's output.

**Where convolution clearly wins is convergence speed.** The hybrids stopped
after 9–10 epochs; the pure recurrent models needed 16–19 to reach a worse
result. Local features hand the recurrent layer a shorter, cleaner sequence
(5 steps instead of 28) and it settles much faster.

**Bidirectionality barely matters here.** BiLSTM is *worse* than LSTM in the
pure setting (−0.08 points for 61% more parameters) and adds only 0.14 points
over CNN+LSTM for 71% more. A character image has no meaningful "future
context" the way a sentence does, so reading rows backwards adds little.

**All five land in the same narrow band.** Despite a 6× spread in capacity,
every model scores between 86.3% and 87.6%. EMNIST-Balanced contains genuinely
ambiguous glyphs — handwritten `O`/`0`, `I`/`l`/`1`, `S`/`5` are often
indistinguishable without context — so a good share of the remaining error is
likely irreducible from a single 28×28 image. Closing that gap probably needs
better data or augmentation more than it needs a bigger encoder.

---

## Dataset

[EMNIST-Balanced](https://www.nist.gov/itl/products-and-services/emnist-dataset):
131,600 handwritten characters across **47 balanced classes** (10 digits, 26
uppercase letters, and 11 lowercase letters whose shapes differ enough from
their uppercase forms to be worth separating).

| Split | Images  |
| :---- | ------: |
| Train | 112,800 |
| Test  |  18,800 |

The CSV distribution stores each image **transposed**, so preprocessing mirrors
and rotates every sample before anything else touches it — see
[`deskew()`](emnist_rnn/data.py). Pixels are then scaled to `[0, 1]` and labels
one-hot encoded.

```
raw CSV row (784 px)  →  reshape 28×28  →  fliplr + rot90  →  /255  →  (28, 28, 1)
```

---

## Quickstart

```bash
git clone <this-repo> && cd Mini_Project_Results
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download the EMNIST CSVs (e.g. from the
[Kaggle mirror](https://www.kaggle.com/datasets/crawford/emnist)) and place them
so the tree looks like this:

```
data/emnist/
├── emnist-balanced-train.csv
├── emnist-balanced-test.csv
└── emnist-balanced-mapping.txt
```

Train one model, or reproduce the whole table:

```bash
python -m emnist_rnn.train --arch cnn_bilstm
python -m emnist_rnn.train --arch all --epochs 30
```

Each run writes a checkpoint to `results/<arch>.keras`, its epoch-by-epoch
history to `results/<arch>_history.json`, and a row into `results/results.csv`.

Score a checkpoint, with an optional per-character breakdown:

```bash
python -m emnist_rnn.evaluate --model results/cnn_bilstm.keras --report
```

Useful flags: `--optimizer {adam,adagrad,adadelta,rmsprop}`, `--batch-size`,
`--patience`, `--data-dir`.

---

## Project structure

```
.
├── emnist_rnn/              # reusable package — the notebooks' logic, deduplicated
│   ├── data.py              # loading, deskewing, normalisation, label map
│   ├── models.py            # the five architectures, one build_model() call
│   ├── train.py             # training CLI + leaderboard writer
│   └── evaluate.py          # scoring CLI + classification report
├── notebooks/               # the original exploratory runs, one per architecture
│   ├── 01_cnn_rnn.ipynb
│   ├── 02_lstm.ipynb
│   ├── 03_bilstm.ipynb
│   ├── 04_cnn_lstm.ipynb
│   └── 05_cnn_bilstm.ipynb
├── results/results.csv      # the table above, machine-readable
├── tests/                   # shape and capacity checks
└── requirements.txt
```

The notebooks are kept for their plots and cell-by-cell narrative, with their
original outputs intact; only the imports and file paths were updated for
Keras 3 and for the local `data/` layout. The package is what you should run
and extend. Its parameter counts are pinned by
[`tests/test_models.py`](tests/test_models.py) to the values the notebooks
produced, so the two can't silently drift apart.

```bash
pytest
```

> **A note on naming.** The original notebook called `RNN` is a *CNN +
> SimpleRNN* hybrid, not a bare RNN — confirmed against its 166,559 parameter
> count. It is named `cnn_rnn` here to match what it actually is.

---

## Method notes

* **Split.** 10% of the training set is held out as a validation set
  (`random_state=88`); Keras carves a further 10% out of the remainder for
  in-training monitoring. The 18,800-image test set is touched only at the end.
* **Stopping.** Early stopping on `val_loss` with patience 5, alongside a
  best-checkpoint callback, so the reported scores come from the best epoch
  rather than the last. None of the five ran the full 30-epoch budget.
* **Fairness.** All five share the same classifier head, optimiser, batch size,
  split seed and callbacks. The recurrent width is fixed at 64 units
  throughout, so parameter counts differ only because the encoders differ.

### Possible extensions

* Attention over the row sequence, to see which strokes drive each decision.
* Per-class confusion analysis on the known-ambiguous pairs (`O`/`0`, `I`/`l`).
* A plain CNN baseline, to isolate the recurrent layer's contribution from the
  other direction.
* Data augmentation (small rotations, elastic distortion) against the ~88%
  ceiling.

---

## License

MIT — see [LICENSE](LICENSE).
