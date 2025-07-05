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
    V.Affine,
    V.Scale,
    V.Translate,
    V.AxialPlaneAffine,
    V.AxialPlaneScale,
    V.AxialPlaneTranslate,
    V.AxialPlaneRotate,
    V.GaussNoise,
    V.GaussBlur,
    V.IntensityShift,
    V.IntensityScale,
    V.Contrast,
    # V.GridDistort,
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
    V.AxialPlaneAffine: {'scale': (0.95, 0.95), 'angle': 5., 'shear': (0., 0.), 'shift': 0.},
}

@pytest.fixture(scope='function', autouse=True)
def seed():
    np.random.seed(1996)

@pytest.fixture(scope='function', autouse=True)
def random(monkeypatch):
    monkeypatch.setattr(np.random, 'random', lambda size=None: np.array([0.25] * size) if size else 0.25)

@pytest.mark.parametrize('transform', SHAPE_PRESERVED_TRANSFORMS + SHAPE_UNPRESERVED_TRANSFORMS)
def test_Transform_CASE_repr(transform):
    instance = transform(always_apply=True)

    repr = str(instance)

    assert 'always_apply' in repr
    assert 'p' in repr

@pytest.mark.parametrize('transform', SHAPE_PRESERVED_TRANSFORMS)
def test_Transform_CASE_call_AND_mono_channel(transform):
    voxel = np.random.randn(32, 32, 32).astype(np.float32)
    mask = np.random.randint(0, 255, size=(32, 32, 32), dtype=np.uint8)
    points = np.random.rand(24, C.NUM_COORDS).astype(np.float32)

    tvoxel = np.copy(voxel)
    tmask = np.copy(mask)
    tpoints = np.copy(points)

    instance = transform(always_apply=True)
    transformed = instance(voxel=tvoxel, mask=tmask, points=tpoints)

    tvoxel, tmask, tpoints = transformed['voxel'], transformed['mask'], transformed['points']

    assert instance.keys() == instance.targets.keys()

    assert tvoxel.flags['C_CONTIGUOUS'] == True
    assert tvoxel.dtype == voxel.dtype
    assert tvoxel.shape == voxel.shape
    assert not np.allclose(tvoxel, voxel)

    assert tmask.flags['C_CONTIGUOUS'] == True
    assert tmask.dtype == mask.dtype
    assert tmask.shape == mask.shape

    if isinstance(instance, V.VoxelOnlyAugmentation):
        assert np.all(tmask == mask)
    else:
        assert not np.all(tmask == mask)

    assert tpoints.flags['C_CONTIGUOUS'] == True
    assert tpoints.dtype == points.dtype
    assert tpoints.shape == points.shape

    if isinstance(instance, V.VoxelOnlyAugmentation):
        assert np.allclose(tpoints, points)
    else:
        assert not np.allclose(tpoints, points)

@pytest.mark.parametrize('transform', SHAPE_PRESERVED_TRANSFORMS)
def test_Transform_CASE_call_AND_multi_channel(transform):
    voxel = np.random.randn(32, 32, 32, 2).astype(np.float32)
    mask = np.random.randint(0, 255, size=(32, 32, 32, 1), dtype=np.uint8)
    points = np.random.rand(24, C.NUM_COORDS).astype(np.float32)

    tvoxel = np.copy(voxel)
    tmask = np.copy(mask)
    tpoints = np.copy(points)

    instance = transform(always_apply=True)
    transformed = instance(voxel=tvoxel, mask=tmask, points=tpoints)

    tvoxel, tmask, tpoints = transformed['voxel'], transformed['mask'], transformed['points']

    assert instance.keys() == instance.targets.keys()

    assert tvoxel.flags['C_CONTIGUOUS'] == True
    assert tvoxel.dtype == voxel.dtype
    assert tvoxel.shape == voxel.shape
    assert not np.allclose(tvoxel, voxel)

    assert tmask.flags['C_CONTIGUOUS'] == True
    assert tmask.dtype == mask.dtype
    assert tmask.shape == mask.shape

    if isinstance(instance, V.VoxelOnlyAugmentation):
        assert np.all(tmask == mask)
    else:
        assert not np.all(tmask == mask)

    assert tpoints.flags['C_CONTIGUOUS'] == True
    assert tpoints.dtype == points.dtype
    assert tpoints.shape == points.shape

    if isinstance(instance, V.VoxelOnlyAugmentation):
        assert np.allclose(tpoints, points)
    else:
        assert not np.allclose(tpoints, points)

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

@pytest.mark.parametrize('transform', SHAPE_PRESERVED_TRANSFORMS + SHAPE_UNPRESERVED_TRANSFORMS)
def test_Transform_CASE_to_dict_AND_from_load_AND_mono_channel(transform):
    SEED = 1996

    input = np.random.randn(32, 32, 32)

    instance = transform(always_apply=True)

    state_dict = instance.to_dict()
    deserialized_instance = V.from_dict(state_dict)

    np.random.seed(SEED)
    output = instance(voxel=input)['voxel']

    np.random.seed(SEED)
    deserialized_output = deserialized_instance(voxel=input)['voxel']

    assert np.allclose(deserialized_output, output)

@pytest.mark.parametrize('transform', SHAPE_PRESERVED_TRANSFORMS + SHAPE_UNPRESERVED_TRANSFORMS)
def test_Transform_CASE_to_dict_AND_from_load_AND_multi_channel(transform):
    SEED = 1996

    input = np.random.randn(32, 32, 32, 2)

    instance = transform(always_apply=True)

    state_dict = instance.to_dict()
    deserialized_instance = V.from_dict(state_dict)

    np.random.seed(SEED)
    output = instance(voxel=input)['voxel']

    np.random.seed(SEED)
    deserialized_output = deserialized_instance(voxel=input)['voxel']

    assert np.allclose(deserialized_output, output)
