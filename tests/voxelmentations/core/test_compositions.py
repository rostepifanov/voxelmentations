import pytest

import numpy as np
import voxelmentations as V

@pytest.mark.core
def test_Sequential_CASE_create_AND_list_error():
    with pytest.raises(RuntimeError, match=r'transforms is type of <.+> that is not list'):
        transform = V.Sequential(
            V.AxialFlip(always_apply=True),
        )

@pytest.mark.core
def test_Sequential_CASE_create_AND_subtype_error():
    with pytest.raises(RuntimeError, match=r'object at \d+ position is not subtype of Apply'):
        transform = V.Sequential([
            V.AxialFlip(always_apply=True),
            object(),
        ], always_apply=True)

@pytest.mark.core
def test_Sequential_CASE_call_AND_no_transfroms():
    input = np.random.randn(32, 32, 32, 1)

    transform = V.Sequential([
    ], always_apply=True)

    output = transform(voxel=input)['voxel']

    assert np.allclose(output, input)

@pytest.mark.core
def test_Sequential_CASE_call_AND_one_flip():
    input = np.random.randn(32, 32, 32, 1)

    transform = V.Sequential([
        V.AxialFlip(always_apply=True),
    ], always_apply=True)

    output = transform(voxel=input)['voxel']

    assert not np.allclose(output, input)

@pytest.mark.core
def test_Sequential_CASE_call_AND_double_flip():
    input = np.random.randn(32, 32, 32, 1)

    transform = V.Sequential([
        V.AxialFlip(always_apply=True),
        V.AxialFlip(always_apply=True),
    ], always_apply=True)

    output = transform(voxel=input)['voxel']

    assert np.allclose(output, input)

@pytest.mark.core
def test_Sequential_CASE_additional_targets():
    input = np.random.randn(32, 32, 32, 1)

    transform = V.Sequential([
        V.AxialFlip(always_apply=True),
    ], always_apply=True)

    transform.add_targets({'voxel2': 'voxel'})

    outputs = transform(voxel=input, voxel2=input)
    output = outputs['voxel2']
    expected = outputs['voxel']

    assert np.allclose(output, expected)

@pytest.mark.core
def test_NonSequential_CASE_additional_targets():
    input = np.random.randn(32, 32, 32, 1)

    transform = V.NonSequential([
        V.AxialFlip(always_apply=True),
    ], always_apply=True)

    transform.add_targets({'voxel2': 'voxel'})

    outputs = transform(voxel=input, voxel2=input)
    output = outputs['voxel2']
    expected = outputs['voxel']

    assert np.allclose(output, expected)

@pytest.mark.core
def test_OneOf_CASE_call_AND_no_transfroms():
    input = np.random.randn(32, 32, 32, 1)

    transform = V.OneOf([
    ], always_apply=True)

    output = transform(voxel=input)['voxel']

    assert np.allclose(output, input)

@pytest.mark.core
def test_OneOf_CASE_call_AND_check_application():
    input = np.random.randn(32, 32, 32, 1)

    transform = V.Sequential([
        V.AxialFlip(always_apply=True),
        V.OneOf([
            V.AxialFlip(),
            V.AxialFlip(),
        ], always_apply=True),
    ], always_apply=True)

    output = transform(voxel=input)['voxel']

    assert np.allclose(output, input)
