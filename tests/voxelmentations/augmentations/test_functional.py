import pytest

import numpy as np

import voxelmentations.core.enum as E
import voxelmentations.augmentations.functional as F

def test_plane_dropout_CASE_zero_dim():
    input = np.expand_dims(
        np.array([1, 2, 3, 4]),
        axis=(1, 2, 3)
    )

    expected = np.expand_dims(
        np.array([0, 0, 3, 4]),
        axis=(1, 2, 3)
    )

    indices = [0, 1]
    fill_value = 0
    dim = 0

    output = F.plane_dropout(input, indices, fill_value, dim)

    assert np.allclose(output, expected)

def test_pad_CASE_left_AND_constant_border():
    input = np.expand_dims(
        np.array([1, 2, 3, 4]),
        axis=(1, 2, 3)
    )

    expected = np.expand_dims(
        np.array([1, 2, 3, 4, 0, 0]),
        axis=(1, 2, 3)
    )

    pads = ((0, 2), (0, 0), (0, 0))
    border_mode = E.BorderType.CONSTANT
    fill_value = 0

    output = F.pad(input, pads, border_mode, fill_value)

    assert np.allclose(output, expected)

def test_pad_CASE_left_AND_constant_border_AND_mono_channel():
    input = np.expand_dims(
        np.array([1, 2, 3, 4]),
        axis=(1, 2)
    )

    expected = np.expand_dims(
        np.array([1, 2, 3, 4, 0, 0]),
        axis=(1, 2)
    )

    pads = ((0, 2), (0, 0), (0, 0))
    border_mode = E.BorderType.CONSTANT
    fill_value = 0

    output = F.pad(input, pads, border_mode, fill_value)

    assert np.allclose(output, expected)

def test_pad_CASE_left_AND_replicate_border():
    input = np.expand_dims(
        np.array([1, 2, 3, 4]),
        axis=(1, 2, 3)
    )

    expected = np.expand_dims(
        np.array([1, 2, 3, 4, 4, 4]),
        axis=(1, 2, 3)
    )

    pads = ((0, 2), (0, 0), (0, 0))
    border_mode = E.BorderType.REPLICATE
    fill_value = None

    output = F.pad(input, pads, border_mode, fill_value)

    assert np.allclose(output, expected)
