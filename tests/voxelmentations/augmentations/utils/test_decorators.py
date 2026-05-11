import pytest

import numpy as np

import voxelmentations.augmentations.utils.decorators as D

def test_contiguous_CASE_non_contiguous_output():
    input = np.random.randn(8, 8, 8, 1)

    @D.contiguous
    def flip_view(input):
        return np.flip(input, axis=0)

    output = flip_view(input)

    assert output.flags['C_CONTIGUOUS'] == True

def test_preserve_channel_dim_CASE_multi_channel_AND_dummy_dim():
    input = np.random.randn(32, 32, 32, 1)

    @D.preserve_channel_dim
    def squeeze(input):
        return input[:, :, :, 0]

    output = squeeze(input)

    assert output.shape == input.shape
