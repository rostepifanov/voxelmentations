import pytest

import numpy as np
import voxelmentations as V
import voxelmentations.core.constants as C

SHAPE_PRESERVED_TRANSFORMS = [
    V.Flip,
    V.AxialFlip,
    V.AxialPlaneFlip,
    V.PlaneDropout,
    V.HorizontalPlaneDropout,
    V.VerticalPlaneDropout,
    V.AxialPlaneDropout,
    V.AxialPlaneRotate,
    V.AxialPlaneScale,
    V.AxialPlaneAffine,
    V.GaussNoise,
    V.GaussBlur,
    V.IntensityShift,
    V.GridDistort,
]

SHAPE_UNPRESERVED_TRANSFORMS = [
    V.PadIfNeeded,
]

MASK_FILL_VALUE_TRANSFORMS = {
    V.PadIfNeeded: {'pads': ((1, 1), (1, 1), (1, 1))},
    V.PlaneDropout: {'indices': [0], 'dim': C.AXIAL_DIM},
    V.HorizontalPlaneDropout: {'indices': [0], 'dim': C.HORIZONTAL_DIM},
    V.VerticalPlaneDropout: {'indices': [0], 'dim': C.VERTICAL_DIM},
    V.AxialPlaneDropout: {'indices': [0], 'dim': C.AXIAL_DIM},
    V.AxialPlaneRotate: {'angle': 5.},
    V.AxialPlaneScale: {'scale': 0.95},
    V.AxialPlaneAffine: {'angle': 5., 'shift': 0., 'scale': 0.95},
}

@pytest.mark.parametrize('transform', SHAPE_PRESERVED_TRANSFORMS + SHAPE_UNPRESERVED_TRANSFORMS)
def test_Transform_CASE_repr(transform):
    instance = transform(always_apply=True)

    repr = str(instance)

    assert 'always_apply' in repr
    assert 'p' in repr

@pytest.mark.parametrize('transform', SHAPE_PRESERVED_TRANSFORMS)
def test_Transform_CASE_call_AND_multi_channel(transform):
    voxel = np.random.randn(32, 32, 32, 2)
    mask = np.ones((32, 32, 32, 1))

    instance = transform(always_apply=True)
    transformed = instance(voxel=voxel, mask=mask)

    tvoxel, tmask = transformed['voxel'], transformed['mask']

    assert tvoxel.shape == voxel.shape
    assert not np.allclose(tvoxel, voxel)

    assert tmask.shape == mask.shape

    if isinstance(transform, V.VoxelOnlyTransform):
        assert np.all(tmask == mask)

@pytest.mark.parametrize('transform', SHAPE_PRESERVED_TRANSFORMS)
def test_Transform_CASE_call_AND_mono_channel(transform):
    voxel = np.random.randn(32, 32, 32)
    mask = np.ones((32, 32, 32))

    instance = transform(always_apply=True)
    transformed = instance(voxel=voxel, mask=mask)

    tvoxel, tmask = transformed['voxel'], transformed['mask']

    assert tvoxel.shape == voxel.shape
    assert not np.allclose(tvoxel, voxel)

    assert tmask.shape == mask.shape

    if isinstance(transform, V.VoxelOnlyTransform):
        assert np.all(tmask == mask)

@pytest.mark.parametrize('transform', MASK_FILL_VALUE_TRANSFORMS.keys())
def test_Transform_CASE_mask_fill_value(transform, monkeypatch):
    monkeypatch.setattr(transform, 'get_params', lambda self: MASK_FILL_VALUE_TRANSFORMS[transform])
    monkeypatch.setattr(transform, 'get_params_dependent_on_targets', lambda self, params: MASK_FILL_VALUE_TRANSFORMS[transform])

    voxel = np.random.randn(32, 32, 32, 1)
    mask = np.zeros((32, 32, 32, 1))

    base_value = 0
    mask_fill_value = 10
    instance = transform(always_apply=True, mask_fill_value=mask_fill_value)
    transformed = instance(voxel=voxel, mask=mask)

    unique_values_mask = np.unique(transformed['mask'])
    assert set(unique_values_mask) == { base_value, mask_fill_value }
