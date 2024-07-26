import pytest

import numpy as np
import voxelmentations as V

SHAPE_PRESERVED_TRANSFORMS = [
    V.Flip,
    V.AxialFlip,
    V.AxialPlaneFlip,
    V.AxialPlaneDropout,
    V.AxialPlaneRotate,
    V.AxialPlaneScale,
]

SHAPE_UNPRESERVED_TRANSFORMS = [
]

@pytest.mark.parametrize('transform', SHAPE_PRESERVED_TRANSFORMS + SHAPE_UNPRESERVED_TRANSFORMS)
def test_Transform_CASE_repr(transform):
    transform = transform(always_apply=True)

    repr = str(transform)

    assert 'always_apply' in repr
    assert 'p' in repr

@pytest.mark.parametrize('transform', SHAPE_PRESERVED_TRANSFORMS)
def test_Transform_CASE_call(transform):
    voxel = np.random.randn(32, 32, 32, 1)
    mask = np.ones((32, 32, 32, 1))

    transform = transform(always_apply=True)
    transformed = transform(voxel=voxel, mask=mask)

    tvoxel, tmask = transformed['voxel'], transformed['mask']

    assert tvoxel.shape == voxel.shape
    assert not np.allclose(tvoxel, voxel)

    assert tmask.shape == mask.shape

    if isinstance(transform, V.VoxelOnlyTransform):
        assert np.all(tmask == mask)
