"""Tests for the preprocessing helpers that do not need the dataset on disk."""

import numpy as np

from emnist_rnn.data import IMAGE_SIZE, deskew


def test_deskew_transposes_the_stored_image():
    flat = np.arange(IMAGE_SIZE * IMAGE_SIZE)
    upright = deskew(flat)

    assert upright.shape == (IMAGE_SIZE, IMAGE_SIZE)
    # The EMNIST CSV stores images transposed; mirroring then rotating undoes it.
    np.testing.assert_array_equal(upright, flat.reshape(IMAGE_SIZE, IMAGE_SIZE).T)
