import pytest

import numpy as np

import voxelmentations.core.enum as E
import voxelmentations.core.constants as C
import voxelmentations.augmentations.functional.volumetrical as FV

@pytest.mark.functional
def test_plane_dropout_CASE_zero_dim():
    input = np.expand_dims(
        np.array([1, 2, 3, 4]),
        axis=(1, 2, 3)
    )

    expected = np.expand_dims(
        np.array([0, 0, 3, 4]),
        axis=(1, 2, 3)
    )

    indices = [0, 1]
    fill_value = 0
    dim = C.VERTICAL_DIM

    output = FV.plane_dropout(input, indices, fill_value, dim)

    assert np.allclose(output, expected)

@pytest.mark.functional
def test_pad_CASE_left_AND_constant_border():
    input = np.expand_dims(
        np.array([1, 2, 3, 4]),
        axis=(1, 2, 3)
    )

    expected = np.expand_dims(
        np.array([1, 2, 3, 4, 0, 0]),
        axis=(1, 2, 3)
    )

    pads = ((0, 2), (0, 0), (0, 0))
    border_mode = E.BorderType.CONSTANT
    fill_value = 0

    output = FV.pad(input, pads, border_mode, fill_value)

    assert np.allclose(output, expected)

@pytest.mark.functional
def test_pad_CASE_left_AND_constant_border_AND_mono_channel():
    input = np.expand_dims(
        np.array([1, 2, 3, 4]),
        axis=(1, 2)
    )

    expected = np.expand_dims(
        np.array([1, 2, 3, 4, 0, 0]),
        axis=(1, 2)
    )

    pads = ((0, 2), (0, 0), (0, 0))
    border_mode = E.BorderType.CONSTANT
    fill_value = 0

    output = FV.pad(input, pads, border_mode, fill_value)

    assert np.allclose(output, expected)

@pytest.mark.functional
def test_pad_CASE_left_AND_replicate_border():
    input = np.expand_dims(
        np.array([1, 2, 3, 4]),
        axis=(1, 2, 3)
    )

    expected = np.expand_dims(
        np.array([1, 2, 3, 4, 4, 4]),
        axis=(1, 2, 3)
    )

    pads = ((0, 2), (0, 0), (0, 0))
    border_mode = E.BorderType.REPLICATE
    fill_value = None

    output = FV.pad(input, pads, border_mode, fill_value)

    assert np.allclose(output, expected)

@pytest.mark.functional
def test_plane_affine_CASE_90_degree_rotation_AND_square_shape():
    input = np.expand_dims(
        np.array([
            [ 1,  2,  3,  4],
            [ 5,  6,  7,  8],
            [ 9, 10, 11, 12],
            [13, 14, 15, 16],
        ]),
        axis=(2, 3)
    )

    expected = np.expand_dims(
        np.array([
            [13,  9,  5,  1],
            [14, 10,  6,  2],
            [15, 11,  7,  3],
            [16, 12,  8,  4],
        ]),
        axis=(2, 3)
    )

    scale = 1.
    shift = 0.
    angle = 90
    interpolation = E.InterType.NEAREST
    border_mode = E.BorderType.CONSTANT
    fill_value = 0
    dim = C.AXIAL_DIM

    output = FV.plane_affine(
        input,
        scale,
        shift,
        angle,
        interpolation,
        border_mode,
        fill_value,
        dim
    )

    assert np.allclose(output, expected)

@pytest.mark.functional
def test_plane_affine_CASE_90_degree_rotation_AND_rectangle_shape():
    input = np.expand_dims(
        np.array([
            [ 1,  2],
            [ 5,  6],
            [ 9, 10],
            [13, 14],
        ]),
        axis=(2, 3)
    )

    expected = np.expand_dims(
        np.array([
            [ 0, 0],
            [ 9, 5],
            [10, 6],
            [ 0, 0],
        ]),
        axis=(2, 3)
    )

    scale = 1.
    shift = 0.
    angle = 90.
    interpolation = E.InterType.NEAREST
    border_mode = E.BorderType.CONSTANT
    fill_value = 0
    dim = C.AXIAL_DIM

    output = FV.plane_affine(
        input,
        scale,
        shift,
        angle,
        interpolation,
        border_mode,
        fill_value,
        dim
    )

    assert np.allclose(output, expected)

@pytest.mark.functional
def test_plane_scale_CASE_twice_isotropic_upscaling():
    input = np.expand_dims(
        np.array([
            [ 1,  2,  3,  4],
            [ 5,  6,  7,  8],
            [ 9, 10, 11, 12],
            [13, 14, 15, 16],
        ]),
        axis=(2, 3)
    )

    expected = np.expand_dims(
        np.array([
            [ 6,  6,  7,  7],
            [ 6,  6,  7,  7],
            [10, 10, 11, 11],
            [10, 10, 11, 11],
        ]),
        axis=(2, 3)
    )

    scale = 2.
    shift = 0.
    angle = 0.
    interpolation = E.InterType.NEAREST
    border_mode = E.BorderType.CONSTANT
    fill_value = 0
    dim = C.AXIAL_DIM

    output = FV.plane_affine(
        input,
        scale,
        shift,
        angle,
        interpolation,
        border_mode,
        fill_value,
        dim
    )

    assert np.allclose(output, expected)

@pytest.mark.functional
def test_plane_affine_CASE_only_shift():
    input = np.expand_dims(
        np.array([
            [ 1,  2,  3,  4],
            [ 5,  6,  7,  8],
            [ 9, 10, 11, 12],
            [13, 14, 15, 16],
        ]),
        axis=(2, 3)
    )

    expected = np.expand_dims(
        np.array([
            [ 0,  0,  0,  0],
            [ 0,  1,  2,  3],
            [ 0,  5,  6,  7],
            [ 0,  9, 10, 11],
        ]),
        axis=(2, 3)
    )

    angle = 0
    shift = 0.25
    scale = 1
    interpolation = E.InterType.NEAREST
    border_mode = E.BorderType.CONSTANT
    fill_value = 0
    dim = C.AXIAL_DIM

    output = FV.plane_affine(
        input,
        scale,
        shift,
        angle,
        interpolation,
        border_mode,
        fill_value,
        dim
    )

    assert np.allclose(output, expected)

@pytest.mark.functional
def test_plane_affine_CASE_rotation_AND_shift():
    input = np.expand_dims(
        np.array([
            [ 1,  2,  3,  4],
            [ 5,  6,  7,  8],
            [ 9, 10, 11, 12],
            [13, 14, 15, 16],
        ]),
        axis=(2, 3)
    )

    expected = np.expand_dims(
        np.array([
            [  0, 0, 0, 0 ],
            [  9, 5, 1, 0 ],
            [ 10, 6, 2, 0 ],
            [ 11, 7, 3, 0 ],
        ]),
        axis=(2, 3)
    )

    scale = 1.
    shift = 0.25
    angle = 90.
    interpolation = E.InterType.NEAREST
    border_mode = E.BorderType.CONSTANT
    fill_value = 0
    dim = C.AXIAL_DIM

    output = FV.plane_affine(
        input,
        scale,
        shift,
        angle,
        interpolation,
        border_mode,
        fill_value,
        dim
    )

    assert np.allclose(output, expected)

@pytest.mark.functional
def test_distort_CASE_identity():
    input = np.expand_dims(
        np.array([1, 2, 3, 4]),
        axis=(1, 2, 3)
    )

    expected = np.expand_dims(
        np.array([1, 2, 3, 4]),
        axis=(1, 2, 3)
    )

    shape = np.array(input.shape[:C.NUM_SPATIAL_DIMENSIONS])

    func = lambda size: np.linspace(0., size-1, size)
    points = map(func, shape)

    grid = np.meshgrid(*points, indexing='ij')

    output = FV.distort(input, grid, E.InterType.LINEAR)

    assert np.allclose(output, expected)

@pytest.mark.functional
def test_reshape_CASE_twice_reshape():
    shape = (2, 2, 1)
    nshape = (4, 4, 1)

    input = np.expand_dims(
        np.array([
            [ 1, 3],
            [ 5, 7],
        ]),
        axis=(2, 3)
    )

    expected1 = np.expand_dims(
        np.array([
            [ 1, 1, 3, 3],
            [ 1, 1, 3, 3],
            [ 5, 5, 7, 7],
            [ 5, 5, 7, 7],
        ]),
        axis=(2, 3)
    )

    expected2 = np.copy(input)

    output = FV.reshape(input, nshape, E.InterType.NEAREST)
    assert np.allclose(output, expected1)

    output = FV.reshape(output, shape, E.InterType.NEAREST)
    assert np.allclose(output, expected2)

@pytest.mark.functional
def test_resсale_CASE_upsсale():
    scale = (2, 2, 2)

    input = np.expand_dims(
        np.array([
            [ 1, 3],
            [ 5, 7],
        ]),
        axis=(2, 3)
    )

    expected = np.expand_dims(
        np.array([
            [ 1, 1, 3, 3],
            [ 1, 1, 3, 3],
            [ 5, 5, 7, 7],
            [ 5, 5, 7, 7],
        ]),
        axis=(2, 3)
    )

    output = FV.rescale(input, scale, E.InterType.NEAREST)
    assert np.allclose(output, expected)
