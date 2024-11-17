import pytest

import numpy as np

import voxelmentations.core.enum as E
import voxelmentations.core.constants as C

from voxelmentations.augmentations.functional import FV, FG

@pytest.mark.functional
def test_flip_CASE_inside_points():
    input = np.array([
        [1, 2, 3, 1],
        [3, 5, 2, 1],
    ])

    expected = np.array([
        [7, 2, 3, 1],
        [5, 5, 2, 1],
    ])

    dims = [0, ]
    shape = (8, 8, 8)

    output = FG.flip(input, dims, shape)

    assert np.allclose(output, expected)

@pytest.mark.functional
def test_flip_CASE_outside_points():
    input = np.array([
        [-4, 2, 3, 1],
        [10, 5, 2, 1],
    ])

    expected = np.array([
        [12, 2, 3, 1],
        [-2, 5, 2, 1],
    ])

    dims = [0, ]
    shape = (8, 8, 8)

    output = FG.flip(input, dims, shape)

    assert np.allclose(output, expected)

@pytest.mark.functional
def test_transpose_CASE_ordered_dims():
    input = np.array([
        [1, 2, 3, 1],
        [4, 5, 2, 1],
    ])

    expected = np.array([
        [2, 1, 3, 1],
        [5, 4, 2, 1],
    ])

    dims = [0, 1]

    output = FG.transpose(input, dims)

    assert np.allclose(output, expected)

@pytest.mark.functional
def test_transpose_CASE_reversed_dims():
    input = np.array([
        [1, 2, 3, 1],
        [4, 5, 2, 1],
    ])

    expected = np.array([
        [2, 1, 3, 1],
        [5, 4, 2, 1],
    ])

    dims = [1, 0]

    output = FG.transpose(input, dims)

    assert np.allclose(output, expected)

@pytest.mark.functional
def test_plane_affine_CASE_90_degree_rotation_AND_square_shape():
    input = np.array([
        [0.5, 0.5, 0.5, 1.0],
        [2.5, 1.5, 0.5, 1.0],
    ])

    expected = np.array([
        [0.5, 3.5, 0.5, 1.0],
        [1.5, 1.5, 0.5, 1.0],
    ])

    scale = 1.
    shift = 0.
    angle = 90
    dim = C.AXIAL_DIM
    shape = (4, 4, 1, 1)

    output = FG.plane_affine(
        input,
        scale,
        shift,
        angle,
        dim,
        shape,
    )

    assert np.allclose(output, expected)

@pytest.mark.functional
def test_plane_affine_CASE_90_degree_rotation_AND_square_shape_AND_align_with_mask():
    points = np.array([
        [0.5, 0.5, 0.5, 1.0],
    ])

    mask = np.expand_dims(
        np.array([
            [ 1, 0, 0, 0],
            [ 0, 0, 0, 0],
            [ 0, 0, 0, 0],
            [ 0, 0, 0, 0],
        ]),
        axis=(2, 3),
    )

    scale = 1.
    shift = 0.
    angle = 90
    interpolation = E.InterType.NEAREST
    border_mode = E.BorderType.CONSTANT
    fill_value = 0
    dim = C.AXIAL_DIM
    shape = (4, 4, 1, 1)

    tpoints = FG.plane_affine(
        points,
        scale,
        shift,
        angle,
        dim,
        shape,
    )

    tmask = FV.plane_affine(
        mask,
        scale,
        shift,
        angle,
        interpolation,
        border_mode,
        fill_value,
        dim,
    )

    for xidx, yidx, zidx in (tpoints + C.C2I_SHIFT)[:, :C.NUM_SPATIAL_COORDS].astype(np.int32):
        assert tmask[xidx, yidx, zidx] == 1

        tmask[xidx, yidx, zidx] = 0

    assert np.all(tmask == 0)

@pytest.mark.functional
def test_plane_affine_CASE_90_degree_rotation_AND_rectangle_shape():
    input = np.array([
        [1.5, 0.5, 0.5, 1.0],
    ])

    expected = np.array([
        [1.5, 1.5, 0.5, 1.0],
    ])

    scale = 1.
    shift = 0.
    angle = 90.
    dim = C.AXIAL_DIM
    shape = (4, 2, 1, 1)

    output = FG.plane_affine(
        input,
        scale,
        shift,
        angle,
        dim,
        shape,
    )

    assert np.allclose(output, expected)

@pytest.mark.functional
def test_plane_affine_CASE_90_degree_rotation_AND_rectangle_shape_AND_align_with_mask():
    points = np.array([
        [1.5, 0.5, 0.5, 1.0],
    ])

    mask = np.expand_dims(
        np.array([
            [0, 0],
            [1, 0],
            [0, 0],
            [0, 0],
        ]),
        axis=(2, 3),
    )

    scale = 1.
    shift = 0.
    angle = 90.
    interpolation = E.InterType.NEAREST
    border_mode = E.BorderType.CONSTANT
    fill_value = 0
    dim = C.AXIAL_DIM
    shape = (4, 2, 1, 1)

    tpoints = FG.plane_affine(
        points,
        scale,
        shift,
        angle,
        dim,
        shape,
    )

    tmask = FV.plane_affine(
        mask,
        scale,
        shift,
        angle,
        interpolation,
        border_mode,
        fill_value,
        dim,
    )

    for xidx, yidx, zidx in (tpoints + C.C2I_SHIFT)[:, :C.NUM_SPATIAL_COORDS].astype(np.int32):
        assert tmask[xidx, yidx, zidx] == 1

        tmask[xidx, yidx, zidx] = 0

    assert np.all(tmask == 0)

@pytest.mark.functional
def test_plane_scale_CASE_twice_isotropic_upscaling():
    input = np.array([
        [0.5, 0.5, 0.5, 1.0],
        [2.5, 1.5, 0.5, 1.0],
    ])

    expected = np.array([
        [-1.0, -1.0, 0.5, 1.0],
        [ 3.0,  1.0, 0.5, 1.0],
    ])

    scale = 2.
    shift = 0.
    angle = 0.
    dim = C.AXIAL_DIM
    shape = (4, 4, 1, 1)

    output = FG.plane_affine(
        input,
        scale,
        shift,
        angle,
        dim,
        shape,
    )

    assert np.allclose(output, expected)

@pytest.mark.functional
def test_plane_scale_CASE_twice_isotropic_upscaling_AND_align_with_mask():
    points = np.array([
        [1.5, 1.5, 0.5, 1.0],
    ])

    mask = np.expand_dims(
        np.array([
            [ 0, 0, 0, 0],
            [ 0, 1, 0, 0],
            [ 0, 0, 0, 0],
            [ 0, 0, 0, 0],
        ]),
        axis=(2, 3),
    )

    scale = 2.
    shift = 0.
    angle = 0.
    interpolation = E.InterType.NEAREST
    border_mode = E.BorderType.CONSTANT
    fill_value = 0
    dim = C.AXIAL_DIM
    shape = (4, 4, 1, 1)

    tpoints = FG.plane_affine(
        points,
        scale,
        shift,
        angle,
        dim,
        shape,
    )

    tmask = FV.plane_affine(
        mask,
        scale,
        shift,
        angle,
        interpolation,
        border_mode,
        fill_value,
        dim,
    )

    for xidx, yidx, zidx in (tpoints + C.C2I_SHIFT)[:, :C.NUM_SPATIAL_COORDS].astype(np.int32):
        assert tmask[xidx, yidx, zidx] == 1

@pytest.mark.functional
def test_plane_scale_CASE_twice_isotropic_downscaling_AND_align_with_mask():
    points = np.array([
        [1.0, 1.0, 0.5, 1.0],
    ])

    mask = np.expand_dims(
        np.array([
            [ 1, 1, 0, 0],
            [ 1, 1, 0, 0],
            [ 0, 0, 0, 0],
            [ 0, 0, 0, 0],
        ]),
        axis=(2, 3),
    )

    scale = 0.5
    shift = 0.
    angle = 0.
    interpolation = E.InterType.NEAREST
    border_mode = E.BorderType.CONSTANT
    fill_value = 0
    dim = C.AXIAL_DIM
    shape = (4, 4, 1, 1)

    tpoints = FG.plane_affine(
        points,
        scale,
        shift,
        angle,
        dim,
        shape,
    )

    tmask = FV.plane_affine(
        mask,
        scale,
        shift,
        angle,
        interpolation,
        border_mode,
        fill_value,
        dim,
    )

    for xidx, yidx, zidx in (tpoints + C.C2I_SHIFT)[:, :C.NUM_SPATIAL_COORDS].astype(np.int32):
        assert tmask[xidx, yidx, zidx] == 1

        tmask[xidx, yidx, zidx] = 0

    assert np.all(tmask == 0)

@pytest.mark.functional
def test_plane_affine_CASE_only_shift():
    input = np.array([
        [ 0.5, 0.5, 0.5, 1.0],
        [ 2.5, 1.5, 0.5, 1.0],

    ])

    expected = np.array([
        [ 1.5, 1.5, 0.5, 1.0],
        [ 3.5, 2.5, 0.5, 1.0],
    ])

    angle = 0
    shift = 0.25
    scale = 1
    dim = C.AXIAL_DIM
    shape = (4, 4, 1, 1)

    output = FG.plane_affine(
        input,
        scale,
        shift,
        angle,
        dim,
        shape,
    )

    assert np.allclose(output, expected)

@pytest.mark.functional
def test_plane_affine_CASE_only_shift_AND_align_with_mask():
    points = np.array([
        [0.5, 0.5, 0.5, 1],
    ])

    mask = np.expand_dims(
        np.array([
            [ 1, 0, 0, 0],
            [ 0, 0, 0, 0],
            [ 0, 0, 0, 0],
            [ 0, 0, 0, 0],
        ]),
        axis=(2, 3),
    )

    angle = 0
    shift = 0.25
    scale = 1
    interpolation = E.InterType.NEAREST
    border_mode = E.BorderType.CONSTANT
    fill_value = 0
    dim = C.AXIAL_DIM
    shape = (4, 4, 1, 1)

    tpoints = FG.plane_affine(
        points,
        scale,
        shift,
        angle,
        dim,
        shape,
    )

    tmask = FV.plane_affine(
        mask,
        scale,
        shift,
        angle,
        interpolation,
        border_mode,
        fill_value,
        dim,
    )

    for xidx, yidx, zidx in (tpoints + C.C2I_SHIFT)[:, :C.NUM_SPATIAL_COORDS].astype(np.int32):
        assert tmask[xidx, yidx, zidx] == 1

        tmask[xidx, yidx, zidx] = 0

    assert np.all(tmask == 0)
