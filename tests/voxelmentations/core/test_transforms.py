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
def test_Identity_CASE_voxel_only():
    input = np.random.randn(32, 32, 32, 1)
    expected =  np.copy(input)

    transform = V.Identity(always_apply=True)
    output = transform(voxel=input)['voxel']

    assert np.allclose(output, expected)

@pytest.mark.core
def test_Identity_CASE_voxel_AND_mask():
    voxel = np.random.randn(32, 32, 32, 1)
    mask = np.random.randint(0, 255, size=(32, 32, 32))

    tvoxel = np.copy(voxel)
    tmask = np.copy(mask)

    transform = V.Identity(always_apply=True)
    transformed = transform(voxel=tvoxel, mask=tmask)

    tvoxel = transformed['voxel']
    tmask = transformed['mask']

    assert np.allclose(tvoxel, voxel)
    assert np.allclose(tmask, mask)

@pytest.mark.core
def test_Identity_CASE_voxel_AND_points():
    voxel = np.random.randn(32, 32, 32, 1)
    points = np.random.randn(16, 3)

    tvoxel = np.copy(voxel)
    tpoints = np.copy(points)

    transform = V.Identity(always_apply=True)
    transformed = transform(voxel=tvoxel, points=tpoints)

    tvoxel = transformed['voxel']
    tpoints = transformed['points']

    assert np.allclose(tvoxel, voxel)
    assert np.allclose(tpoints, points)

@pytest.mark.core
def test_Transform_CASE_additional_targets():
    input = np.random.randn(32, 32, 32, 1)
    expected = np.copy(input)

    transform = V.Identity(always_apply=True)
    transform.add_targets({'voxel2': 'voxel'})

    output = transform(voxel=input, voxel2=input)['voxel2']

    assert np.allclose(output, expected)

@pytest.mark.core
def test_Transform_CASE_additional_targets_CASE_key_rewrite():
    input = np.random.randn(32, 32, 32, 1)

    transform = V.Identity(always_apply=True)

    with pytest.raises(ValueError):
        transform.add_targets({'voxel': 'voxel'})
