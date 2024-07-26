import pytest

import numpy as np

import voxelmentations.core.functional as F

@pytest.mark.core
def test_preserve_channel_dim_CASE_multi_channel_AND_dummy_dim():
    input = np.random.randn(32, 32, 32, 1)

    @F.preserve_channel_dim
    def squeeze(input):
        return input[:, :, :, 0]

    output = squeeze(input)

    assert output.shape == input.shape
