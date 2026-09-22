"""Shape and capacity checks for the five architectures.

Parameter counts are pinned to the values produced by the original notebook
runs, so an accidental change to a layer width shows up as a test failure
rather than as a silently different result table.
"""

import pytest

from emnist_rnn.data import IMAGE_SIZE, NUM_CLASSES
from emnist_rnn.models import ARCHITECTURES, build_model

EXPECTED_PARAMETERS = {
    "lstm": 66_031,
    "bilstm": 106_223,
    "cnn_rnn": 166_559,
    "cnn_lstm": 225_119,
    "cnn_bilstm": 385_119,
}


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_input_and_output_shapes(architecture):
    model = build_model(architecture)
    assert model.input_shape == (None, IMAGE_SIZE, IMAGE_SIZE, 1)
    assert model.output_shape == (None, NUM_CLASSES)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_parameter_counts_match_the_published_results(architecture):
    assert build_model(architecture).count_params() == EXPECTED_PARAMETERS[architecture]


def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError, match="unknown architecture"):
        build_model("transformer")
