import pytest

import numpy as np
import voxelmentations as V

@pytest.mark.core
def test_Identity_CASE_repr():
    transformation = V.Identity(always_apply=True)

    repr = str(transformation)

    assert 'Identity' in repr
    assert 'always_apply' in repr
    assert 'p' in repr

@pytest.mark.core
def test_Identity_CASE_voxel_only():
    input = np.random.randn(32, 32, 32, 1)
    expected =  np.copy(input)

    transformation = V.Identity(always_apply=True)
    output = transformation(voxel=input)['voxel']

    assert np.allclose(output, expected)

@pytest.mark.core
def test_Identity_CASE_voxel_AND_mask():
    voxel = np.random.randn(32, 32, 32, 1)
    mask = np.random.randint(0, 255, size=(32, 32, 32))

    tvoxel = np.copy(voxel)
    tmask = np.copy(mask)

    transformation = V.Identity(always_apply=True)
    transformed = transformation(voxel=tvoxel, mask=tmask)

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

    transformation = V.Identity(always_apply=True)
    transformed = transformation(voxel=tvoxel, points=tpoints)

    tvoxel = transformed['voxel']
    tpoints = transformed['points']

    assert np.allclose(tvoxel, voxel)
    assert np.allclose(tpoints, points)

@pytest.mark.core
def test_Transform_CASE_additional_targets():
    input = np.random.randn(32, 32, 32, 1)
    expected = np.copy(input)

    transformation = V.Identity(always_apply=True)
    transformation.add_targets({'voxel2': 'voxel'})

    output = transformation(voxel=input, voxel2=input)['voxel2']

    assert np.allclose(output, expected)

@pytest.mark.core
def test_Transform_CASE_additional_targets_CASE_key_rewrite():
    input = np.random.randn(32, 32, 32, 1)

    transformation = V.Identity(always_apply=True)

    with pytest.raises(ValueError):
        transformation.add_targets({'voxel': 'voxel'})