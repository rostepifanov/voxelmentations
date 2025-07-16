import pytest

import numpy as np

import voxelmentations.core.enum as E
import voxelmentations.core.constants as C

from voxelmentations.augmentations.functional import FV, FG

AFFINE_VOLUMETRIC_EQUAL_TO_PLANAR = {
    'scale_AND_axial_dim_AND_downscale_AND_even_shape': {
        'input': np.expand_dims(
            np.array([
                [ 1, 1, 5, 5],
                [ 1, 1, 5, 5],
                [ 3, 3, 7, 7],
                [ 3, 3, 7, 7],
            ]),
            axis=(2, 3),
        ),
        'planar_args': {
            'scale': (0.5, 0.5),
            'angle': 0.,
            'shear': (0., 0.),
            'shift': (0., 0.),
            'dim': C.AXIAL_DIM,
        },
        'volumetric_args': {
            'scale': (0.5, 0.5, 1.),
            'angles': (0., 0., 0.),
            'shift': (0., 0., 0.)
        },
        'interpolation': E.InterType.NEAREST,
        'border_mode': E.BorderType.CONSTANT,
        'fill_value': 0,
    },
    'scale_AND_axial_dim_AND_upscale_AND_odd_shape': {
        'input': np.expand_dims(
            np.array([
                [ 1, 1, 1, 1, 1],
                [ 1, 3, 3, 3, 1],
                [ 1, 3, 5, 3, 1],
                [ 1, 3, 3, 3, 1],
                [ 1, 1, 1, 1, 1],
            ]),
            axis=(2, 3),
        ),
        'planar_args': {
            'scale': (2., 2.),
            'angle': 0.,
            'shear': (0., 0.),
            'shift': (0., 0.),
            'dim': C.AXIAL_DIM,
        },
        'volumetric_args': {
            'scale': (2., 2., 1.),
            'angles': (0., 0., 0.),
            'shift': (0., 0., 0.)
        },
        'interpolation': E.InterType.NEAREST,
        'border_mode': E.BorderType.CONSTANT,
        'fill_value': 0,
    },
    'scale_AND_axial_dim_AND_upscale_AND_even_shape': {
        'input': np.expand_dims(
            np.array([
                [ 1,  2,  3,  4],
                [ 5,  6,  7,  8],
                [ 9, 10, 11, 12],
                [13, 14, 15, 16],
            ]),
            axis=(2, 3),
        ),
        'planar_args': {
            'scale': (2., 2.),
            'angle': 0.,
            'shear': (0., 0.),
            'shift': (0., 0.),
            'dim': C.AXIAL_DIM,
        },
        'volumetric_args': {
            'scale': (2., 2., 1.),
            'angles': (0., 0., 0.),
            'shift': (0., 0., 0.)
        },
        'interpolation': E.InterType.NEAREST,
        'border_mode': E.BorderType.CONSTANT,
        'fill_value': 0,
    },
    'rotate_AND_axial_dim_AND_90deg_AND_even_shape': {
        'input': np.expand_dims(
            np.array([
                [ 1,  2,  3,  4],
                [ 5,  6,  7,  8],
                [ 9, 10, 11, 12],
                [13, 14, 15, 16],
            ]),
            axis=(2, 3),
        ),
        'planar_args': {
            'scale': (1., 1.),
            'angle': 90.,
            'shear': (0., 0.),
            'shift': (0., 0.),
            'dim': C.AXIAL_DIM,
        },
        'volumetric_args': {
            'scale': (1., 1., 1.),
            'angles': (0., 0., 90.),
            'shift': (0., 0., 0.)
        },
        'interpolation': E.InterType.NEAREST,
        'border_mode': E.BorderType.CONSTANT,
        'fill_value': 0,
    },
    'rotate_AND_vertical_dim_AND_90deg_AND_even_shape': {
        'input': np.expand_dims(
            np.array([
                [ 1,  2,  3,  4],
                [ 5,  6,  7,  8],
                [ 9, 10, 11, 12],
                [13, 14, 15, 16],
            ]),
            axis=(0, 3),
        ),
        'planar_args': {
            'scale': (1., 1.),
            'angle': 90.,
            'shear': (0., 0.),
            'shift': (0., 0.),
            'dim': C.VERTICAL_DIM,
        },
        'volumetric_args': {
            'scale': (1., 1., 1.),
            'angles': (90., 0., 0.),
            'shift': (0., 0., 0.)
        },
        'interpolation': E.InterType.NEAREST,
        'border_mode': E.BorderType.CONSTANT,
        'fill_value': 0,
    },
    'translate_AND_axial_dim_AND_hhalfsize_AND_even_shape': {
        'input': np.expand_dims(
            np.array([
                [ 1,  2,  3,  4],
                [ 5,  6,  7,  8],
                [ 9, 10, 11, 12],
                [13, 14, 15, 16],
            ]),
            axis=(2, 3),
        ),
        'planar_args': {
            'scale': (1., 1.),
            'angle': 0.,
            'shear': (0., 0.),
            'shift': (0.25, 0.25),
            'dim': C.AXIAL_DIM,
        },
        'volumetric_args': {
            'scale': (1., 1., 1.),
            'angles': (0., 0., 0.),
            'shift': (0.25, 0.25, 0.)
        },
        'interpolation': E.InterType.NEAREST,
        'border_mode': E.BorderType.CONSTANT,
        'fill_value': 0,
    },
    'affine_AND_axial_dim_AND_90deg_hhalfsize_AND_even_shape': {
        'input': np.expand_dims(
            np.array([
                [ 1,  2,  3,  4],
                [ 5,  6,  7,  8],
                [ 9, 10, 11, 12],
                [13, 14, 15, 16],
            ]),
            axis=(2, 3),
        ),
        'planar_args': {
            'scale': (1., 1.),
            'angle': 90.,
            'shear': (0., 0.),
            'shift': (0.25, 0.25),
            'dim': C.AXIAL_DIM,
        },
        'volumetric_args': {
            'scale': (1., 1., 1.),
            'angles': (0., 0., 90.),
            'shift': (0.25, 0.25, 0.)
        },
        'interpolation': E.InterType.NEAREST,
        'border_mode': E.BorderType.CONSTANT,
        'fill_value': 0,
    },
}

@pytest.mark.functional
def test_plane_dropout_CASE_zero_dim():
    input = np.expand_dims(
        np.array([1, 2, 3, 4]),
        axis=(1, 2, 3),
    )

    expected = np.expand_dims(
        np.array([0, 0, 3, 4]),
        axis=(1, 2, 3),
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
        axis=(1, 2, 3),
    )

    expected = np.expand_dims(
        np.array([1, 2, 3, 4, 0, 0]),
        axis=(1, 2, 3),
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
        axis=(1, 2),
    )

    expected = np.expand_dims(
        np.array([1, 2, 3, 4, 0, 0]),
        axis=(1, 2),
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
        axis=(1, 2, 3),
    )

    expected = np.expand_dims(
        np.array([1, 2, 3, 4, 4, 4]),
        axis=(1, 2, 3),
    )

    pads = ((0, 2), (0, 0), (0, 0))
    border_mode = E.BorderType.REPLICATE
    fill_value = None

    output = FV.pad(input, pads, border_mode, fill_value)

    assert np.allclose(output, expected)

@pytest.mark.functional
@pytest.mark.parametrize('case', AFFINE_VOLUMETRIC_EQUAL_TO_PLANAR.keys())
def test_affine_CASE_volumetric_equal_to_planar(case):
    args = AFFINE_VOLUMETRIC_EQUAL_TO_PLANAR[case]

    volumetric_output = FV.affine(
        args['input'],
        interpolation=args['interpolation'],
        border_mode=args['border_mode'],
        fill_value=args['fill_value'],
        **args['volumetric_args'],
    )

    planar_output = FV.plane_affine(
        args['input'],
        interpolation=args['interpolation'],
        border_mode=args['border_mode'],
        fill_value=args['fill_value'],
        **args['planar_args'],
    )

    assert np.allclose(volumetric_output, planar_output)

@pytest.mark.functional
def test_affine_CASE_planar_equal_to_volumetric_AND_scaling_AND_twice_isotropic_upscaling():
    input = np.expand_dims(
        np.array([
            [
                [ 1,  2,  3,  4],
                [ 5,  6,  7,  8],
                [ 9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
            [
                [17, 18, 19, 20],
                [21, 22, 23, 24],
                [25, 26, 27, 28],
                [29, 30, 31, 32],
            ],
        ]),
        axis=(3, ),
    )

    scale = 2.
    volumetric_scale = (scale, scale, scale)
    planar_scale = (scale, scale)

    interpolation = E.InterType.NEAREST
    border_mode = E.BorderType.CONSTANT
    fill_value = 0

    volumetric_output = FV.affine(
        input,
        volumetric_scale,
        (0., 0., 0.),
        0.,
        interpolation,
        border_mode,
        fill_value,
    )

    planar_output = FV.plane_affine(
        input,
        planar_scale,
        0.,
        (0., 0.),
        0.,
        interpolation,
        border_mode,
        fill_value,
        C.AXIAL_DIM,
    )

    planar_output = FV.plane_affine(
        planar_output,
        (1., scale),
        0.,
        (0., 0.),
        0.,
        interpolation,
        border_mode,
        fill_value,
        C.VERTICAL_DIM,
    )

    assert np.allclose(volumetric_output, planar_output)

@pytest.mark.functional
def test_plane_affine_CASE_twice_isotropic_upscaling():
    input = np.expand_dims(
        np.array([
            [ 1,  2,  3,  4],
            [ 5,  6,  7,  8],
            [ 9, 10, 11, 12],
            [13, 14, 15, 16],
        ]),
        axis=(2, 3),
    )

    expected = np.expand_dims(
        np.array([
            [ 6,  6,  7,  7],
            [ 6,  6,  7,  7],
            [10, 10, 11, 11],
            [10, 10, 11, 11],
        ]),
        axis=(2, 3),
    )

    scale = (2., 2.)
    angle = 0.
    shear = (0., 0.)
    shift = 0.
    interpolation = E.InterType.NEAREST
    border_mode = E.BorderType.CONSTANT
    fill_value = 0
    dim = C.AXIAL_DIM

    output = FV.plane_affine(
        input,
        scale,
        angle,
        shear,
        shift,
        interpolation,
        border_mode,
        fill_value,
        dim,
    )

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
        axis=(2, 3),
    )

    expected = np.expand_dims(
        np.array([
            [4, 8, 12, 16],
            [3, 7, 11, 15],
            [2, 6, 10, 14],
            [1, 5,  9, 13],
        ]),
        axis=(2, 3),
    )

    scale = (1., 1.)
    angle = 90
    shear = (0., 0.)
    shift = 0.
    interpolation = E.InterType.NEAREST
    border_mode = E.BorderType.CONSTANT
    fill_value = 0
    dim = C.AXIAL_DIM

    output = FV.plane_affine(
        input,
        scale,
        angle,
        shear,
        shift,
        interpolation,
        border_mode,
        fill_value,
        dim,
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
        axis=(2, 3),
    )

    expected = np.expand_dims(
        np.array([
            [0,  0],
            [6, 10],
            [5,  9],
            [0,  0],
        ]),
        axis=(2, 3),
    )

    scale = (1., 1.)
    angle = 90.
    shear = (0., 0.)
    shift = 0.
    interpolation = E.InterType.NEAREST
    border_mode = E.BorderType.CONSTANT
    fill_value = 0
    dim = C.AXIAL_DIM

    output = FV.plane_affine(
        input,
        scale,
        angle,
        shear,
        shift,
        interpolation,
        border_mode,
        fill_value,
        dim,
    )

    assert np.allclose(output, expected)

@pytest.mark.functional
def test_plane_affine_CASE_only_translation():
    input = np.expand_dims(
        np.array([
            [ 1,  2,  3,  4],
            [ 5,  6,  7,  8],
            [ 9, 10, 11, 12],
            [13, 14, 15, 16],
        ]),
        axis=(2, 3),
    )

    expected = np.expand_dims(
        np.array([
            [ 0,  0,  0,  0],
            [ 0,  1,  2,  3],
            [ 0,  5,  6,  7],
            [ 0,  9, 10, 11],
        ]),
        axis=(2, 3),
    )

    scale = (1. ,1.)
    angle = 0
    shear = (0., 0.)
    shift = 0.25
    interpolation = E.InterType.NEAREST
    border_mode = E.BorderType.CONSTANT
    fill_value = 0
    dim = C.AXIAL_DIM

    output = FV.plane_affine(
        input,
        scale,
        angle,
        shear,
        shift,
        interpolation,
        border_mode,
        fill_value,
        dim,
    )

    assert np.allclose(output, expected)

@pytest.mark.functional
def test_plane_affine_CASE_rotation_AND_translation():
    input = np.expand_dims(
        np.array([
            [ 1,  2,  3,  4],
            [ 5,  6,  7,  8],
            [ 9, 10, 11, 12],
            [13, 14, 15, 16],
        ]),
        axis=(2, 3),
    )

    expected = np.expand_dims(
        np.array([
            [ 0, 0, 0,  0 ],
            [ 0, 4, 8, 12 ],
            [ 0, 3, 7, 11 ],
            [ 0, 2, 6, 10 ],
        ]),
        axis=(2, 3),
    )

    scale = (1., 1.)
    angle = 90.
    shear = (0., 0.)
    shift = 0.25
    interpolation = E.InterType.NEAREST
    border_mode = E.BorderType.CONSTANT
    fill_value = 0
    dim = C.AXIAL_DIM

    output = FV.plane_affine(
        input,
        scale,
        angle,
        shear,
        shift,
        interpolation,
        border_mode,
        fill_value,
        dim,
    )

    assert np.allclose(output, expected)

@pytest.mark.functional
def test_distort_CASE_identity():
    input = np.expand_dims(
        np.array([1, 2, 3, 4]),
        axis=(1, 2, 3),
    )

    expected = np.expand_dims(
        np.array([1, 2, 3, 4]),
        axis=(1, 2, 3),
    )

    shape = np.array(input.shape[:C.NUM_SPATIAL_DIMENSIONS])

    func = lambda size: np.linspace(0., size-1, size)
    points = map(func, shape)

    grid = np.meshgrid(*points, indexing='ij')

    output = FV.distort(input, grid, E.InterType.LINEAR)

    assert np.allclose(output, expected)

@pytest.mark.functional
def test_contrast_CASE_contrast():
    contrast = 2

    input = np.expand_dims(
        np.array([0, -1, 1]),
        axis=(1, 2, 3),
    )

    expected = np.expand_dims(
        np.array([0, -2, 2]),
        axis=(1, 2, 3),
    )

    output = FV.contrast(input, contrast)

    assert np.allclose(output, expected)

@pytest.mark.functional
def test_gamma_CASE_gamma_square():
    gamma = 2

    input = np.expand_dims(
        np.array([1, -2, 3, 4]),
        axis=(1, 2, 3),
    )

    expected = np.expand_dims(
        np.array([1, -4, 9, 16]),
        axis=(1, 2, 3),
    )

    output = FV.gamma(input, gamma)

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
        axis=(2, 3),
    )

    expected1 = np.expand_dims(
        np.array([
            [ 1, 1, 3, 3],
            [ 1, 1, 3, 3],
            [ 5, 5, 7, 7],
            [ 5, 5, 7, 7],
        ]),
        axis=(2, 3),
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
        axis=(2, 3),
    )

    expected = np.expand_dims(
        np.array([
            [ 1, 1, 3, 3],
            [ 1, 1, 3, 3],
            [ 5, 5, 7, 7],
            [ 5, 5, 7, 7],
        ]),
        axis=(2, 3),
    )

    output = FV.rescale(input, scale, E.InterType.NEAREST)
    assert np.allclose(output, expected)
