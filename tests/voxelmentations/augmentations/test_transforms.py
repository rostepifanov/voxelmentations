import pytest

import numpy as np
import voxelmentations as V
import voxelmentations.core.constants as C

SHAPE_PRESERVED_TRANSFORMS = [
    V.Flip,
    V.AxialFlip,
    V.AxialPlaneFlip,
    V.Rotate90,
    V.AxialPlaneRotate90,
    V.Tranpose,
    V.AxialPlaneTranpose,
    V.AxialPlaneAffine,
    V.AxialPlaneScale,
    V.AxialPlaneTranslate,
    V.AxialPlaneRotate,
    V.GaussNoise,
    V.GaussBlur,
    V.IntensityShift,
    V.IntensityScale,
    V.Contrast,
    V.GridDistort,
    # V.ElasticDistort,
    V.Gamma,
    V.PlaneDropout,
    V.HorizontalPlaneDropout,
    V.VerticalPlaneDropout,
    V.AxialPlaneDropout,
    V.PatchDropout,
    V.PatchShuffle,
    V.Downscale,
]

SHAPE_UNPRESERVED_TRANSFORMS = [
    V.PadIfNeeded,
]

MASK_FILL_VALUE_TRANSFORMS = {
    V.PadIfNeeded: {'pads': ((1, 1), (1, 1), (1, 1))},
    V.AxialPlaneAffine: {'angle': 5., 'shift': 0., 'scale': 0.95},
}

@pytest.fixture(scope='function', autouse=True)
def seed():
    np.random.seed(1996)

@pytest.fixture(scope='function', autouse=True)
def random(monkeypatch):
    monkeypatch.setattr(np.random, 'random', lambda size=None: 0.25)

@pytest.mark.parametrize('transform', SHAPE_PRESERVED_TRANSFORMS + SHAPE_UNPRESERVED_TRANSFORMS)
def test_Transform_CASE_repr(transform):
    instance = transform(always_apply=True)

    repr = str(instance)

    assert 'always_apply' in repr
    assert 'p' in repr

@pytest.mark.parametrize('transform', SHAPE_PRESERVED_TRANSFORMS)
def test_Transform_CASE_call_AND_mono_channel(transform):
    voxel = np.random.randn(32, 32, 32)
    mask = np.ones((32, 32, 32), dtype=np.uint8)
    points = np.ones((24, C.NUM_COORDS), dtype=np.float32)

    tvoxel = np.copy(voxel)
    tmask = np.copy(mask)
    tpoints = np.copy(points)

    instance = transform(always_apply=True)
    transformed = instance(voxel=tvoxel, mask=tmask, points=tpoints)

    tvoxel, tmask, tpoints = transformed['voxel'], transformed['mask'], transformed['points']

    assert tvoxel.shape == voxel.shape
    assert not np.allclose(tvoxel, voxel)

    assert tmask.shape == mask.shape

    if isinstance(transform, V.VoxelOnlyTransform):
        assert np.all(tmask == mask)

    assert tpoints.shape == points.shape

    if isinstance(transform, V.VoxelOnlyTransform):
        assert np.allclose(tpoints, points)

@pytest.mark.parametrize('transform', SHAPE_PRESERVED_TRANSFORMS)
def test_Transform_CASE_call_AND_multi_channel(transform):
    voxel = np.random.randn(32, 32, 32, 2)
    mask = np.ones((32, 32, 32, 1), dtype=np.uint8)
    points = np.ones((24, C.NUM_COORDS), dtype=np.float32)

    tvoxel = np.copy(voxel)
    tmask = np.copy(mask)
    tpoints = np.copy(points)

    instance = transform(always_apply=True)
    transformed = instance(voxel=tvoxel, mask=tmask, points=tpoints)

    tvoxel, tmask, tpoints = transformed['voxel'], transformed['mask'], transformed['points']

    assert tvoxel.shape == voxel.shape
    assert not np.allclose(tvoxel, voxel)

    assert tmask.shape == mask.shape

    if isinstance(transform, V.VoxelOnlyTransform):
        assert np.all(tmask == mask)

    assert tpoints.shape == points.shape

    if isinstance(transform, V.VoxelOnlyTransform):
        assert np.allclose(tpoints, points)

@pytest.mark.parametrize('transform', MASK_FILL_VALUE_TRANSFORMS.keys())
def test_Transform_CASE_mask_fill_value(transform, monkeypatch):
    monkeypatch.setattr(transform, 'get_params', lambda self: MASK_FILL_VALUE_TRANSFORMS[transform])
    monkeypatch.setattr(transform, 'get_params_dependent_on_targets', lambda self, params: MASK_FILL_VALUE_TRANSFORMS[transform])

    voxel = np.random.randn(32, 32, 32, 1)
    mask = np.zeros((32, 32, 32, 1), dtype=np.uint8)

    base_value = 0
    mask_fill_value = 10
    instance = transform(always_apply=True, mask_fill_value=mask_fill_value)
    transformed = instance(voxel=voxel, mask=mask)

    unique_values_mask = np.unique(transformed['mask'])
    assert set(unique_values_mask) == { base_value, mask_fill_value }
