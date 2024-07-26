import pytest

import numpy as np
import voxelmentations as V

@pytest.mark.core
def test_Identity_CASE_repr():
    transform = V.Identity(always_apply=True)

    repr = str(transform)

    assert 'Identity' in repr
    assert 'always_apply' in repr
    assert 'p' in repr

@pytest.mark.core
def test_Identity_CASE_call():
    input = np.random.randn(32, 32, 32, 1)

    transform = V.Identity(always_apply=True)

    output = transform(voxel=input)['voxel']
    expected = input

    assert np.allclose(output, expected)

@pytest.mark.core
def test_Transform_CASE_additional_targets():
    input = np.random.randn(32, 32, 32, 1)

    transform = V.Identity(always_apply=True)
    transform.add_targets({'voxel2': 'voxel'})

    output = transform(voxel=input, voxel2=input)['voxel2']
    expected = input

    assert np.allclose(output, expected)


@pytest.mark.core
def test_Transform_CASE_additional_targets_CASE_key_rewrite():
    input = np.random.randn(32, 32, 32, 1)

    transform = V.Identity(always_apply=True)

    with pytest.raises(ValueError):
        transform.add_targets({'voxel': 'voxel'})
