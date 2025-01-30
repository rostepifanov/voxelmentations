import pytest

import numpy as np
import voxelmentations as V

@pytest.mark.core
def test_Sequential_CASE_create_AND_list_error():
    with pytest.raises(RuntimeError, match=r'transformations is type of <.+> that is not list'):
        transformation = V.Sequential(
            V.AxialFlip(always_apply=True),
        )

@pytest.mark.core
def test_Sequential_CASE_create_AND_subtype_error():
    with pytest.raises(RuntimeError, match=r'object at \d+ position is not subtype of Transformation'):
        transformation = V.Sequential([
            V.AxialFlip(always_apply=True),
            object(),
        ], always_apply=True)

@pytest.mark.core
def test_Sequential_CASE_call_AND_no_transfroms():
    input = np.random.randn(32, 32, 32, 1)

    transformation = V.Sequential([
    ], always_apply=True)

    output = transformation(voxel=input)['voxel']

    assert np.allclose(output, input)

@pytest.mark.core
def test_Sequential_CASE_call_AND_one_flip():
    input = np.random.randn(32, 32, 32, 1)

    transformation = V.Sequential([
        V.AxialFlip(always_apply=True),
    ], always_apply=True)

    output = transformation(voxel=input)['voxel']

    assert not np.allclose(output, input)

@pytest.mark.core
def test_Sequential_CASE_call_AND_double_flip():
    input = np.random.randn(32, 32, 32, 1)

    transformation = V.Sequential([
        V.AxialFlip(always_apply=True),
        V.AxialFlip(always_apply=True),
    ], always_apply=True)

    output = transformation(voxel=input)['voxel']

    assert np.allclose(output, input)

@pytest.mark.core
def test_Sequential_CASE_additional_keys():
    input = np.random.randn(32, 32, 32, 1)

    transformation = V.Sequential([
        V.AxialFlip(always_apply=True),
    ], always_apply=True)

    transformation.add_keys({'voxel2': 'voxel'})

    outputs = transformation(voxel=input, voxel2=input)
    output = outputs['voxel2']
    expected = outputs['voxel']

    assert 'voxel2' in transformation.keys()
    assert np.allclose(output, expected)

@pytest.mark.core
def test_NonSequential_CASE_additional_keys():
    input = np.random.randn(32, 32, 32, 1)

    transformation = V.NonSequential([
        V.AxialFlip(always_apply=True),
    ], always_apply=True)

    transformation.add_keys({'voxel2': 'voxel'})

    outputs = transformation(voxel=input, voxel2=input)
    output = outputs['voxel2']
    expected = outputs['voxel']

    assert 'voxel2' in transformation.keys()
    assert np.allclose(output, expected)

@pytest.mark.core
def test_Compose_CASE_additional_keys_AND_hierarchy():
    input = np.random.randn(32, 32, 32, 1)

    transformation = V.Sequential([
        V.OneOf([
            V.AxialFlip(always_apply=True),
        ], always_apply=True)
    ], always_apply=True)

    transformation.add_keys({'voxel2': 'voxel'})

    outputs = transformation(voxel=input, voxel2=input)
    output = outputs['voxel2']
    expected = outputs['voxel']

    assert 'voxel2' in transformation.keys()
    assert np.allclose(output, expected)

@pytest.mark.core
def test_Compose_CASE_additional_keys_AND_notarget():
    voxel = np.random.randn(32, 32, 32, 2)
    mask = np.ones((32, 32, 32, 1))

    transformation = V.NonSequential([
        V.AxialFlip(always_apply=True),
        V.GaussNoise(always_apply=True)
    ], always_apply=True)

    transformation.add_keys({'mask2': 'mask'})
    transformed = transformation(voxel=voxel, mask=mask, mask2=mask)

    tmask, tmask2 = transformed['mask'], transformed['mask2']

    assert 'mask2' in transformation.keys()
    assert np.allclose(tmask, tmask2)

@pytest.mark.core
def test_OneOf_CASE_call_AND_no_transfroms():
    input = np.random.randn(32, 32, 32, 1)

    transformation = V.OneOf([
    ], always_apply=True)

    output = transformation(voxel=input)['voxel']

    assert np.allclose(output, input)

@pytest.mark.core
def test_OneOf_CASE_call_AND_check_application():
    input = np.random.randn(32, 32, 32, 1)

    transformation = V.Sequential([
        V.AxialFlip(always_apply=True),
        V.OneOf([
            V.AxialFlip(),
            V.AxialFlip(),
        ], always_apply=True),
    ], always_apply=True)

    output = transformation(voxel=input)['voxel']

    assert np.allclose(output, input)
