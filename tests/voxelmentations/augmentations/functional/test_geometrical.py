import pytest

import numpy as np

import voxelmentations.core.enum as E
import voxelmentations.core.constants as C
import voxelmentations.augmentations.functional.geometrical as FG

@pytest.mark.functional
def test_transpose_CASE_ordered_dims():
    input = np.array([
        [1, 2, 3, 1],
        [4, 5, 2, 1]
    ])

    expected = np.array([
        [2, 1, 3, 1],
        [5, 4, 2, 1]
    ])

    dims = [0, 1]

    output = FG.transpose(input, dims)

    assert np.allclose(output, expected)

@pytest.mark.functional
def test_transpose_CASE_reversed_dims():
    input = np.array([
        [1, 2, 3, 1],
        [4, 5, 2, 1]
    ])

    expected = np.array([
        [2, 1, 3, 1],
        [5, 4, 2, 1]
    ])

    dims = [1, 0]

    output = FG.transpose(input, dims)

    assert np.allclose(output, expected)
