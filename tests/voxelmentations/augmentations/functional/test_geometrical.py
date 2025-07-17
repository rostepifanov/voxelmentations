import pytest

import numpy as np

import voxelmentations.core.enum as E
import voxelmentations.core.constants as C

from voxelmentations.augmentations.functional import FV, FG

@pytest.mark.functional
def test_pad_CASE_default():
    input = np.array([
        [1, 2, 3, 1],
        [3, 5, 2, 1],
    ])

    expected = np.array([
        [2, 4, 6, 1],
        [4, 7, 5, 1],
    ])

    pads = (
        (1, 4),
        (2, 5),
        (3, 6),
    )

    output = FG.pad(input, pads)

    assert np.allclose(output, expected)

@pytest.mark.functional
def test_plane_affine_CASE_90_degree_rotation_AND_square_shape():
    input = np.array([
        [0.5, 0.5, 0.5, 1.0],
        [2.5, 1.5, 0.5, 1.0],
    ])

    expected = np.array([
        [3.5, 0.5, 0.5, 1.0],
        [2.5, 2.5, 0.5, 1.0],
    ])

    scale = (1., 1.)
    angle = 90
    shear = (0., 0.)
    shift = 0.
    dim = C.AXIAL_DIM
    shape = (4, 4, 1, 1)

    output = FG.plane_affine(
        input,
        scale,
        angle,
        shear,
        shift,
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

    scale = (1., 1.)
    angle = 90
    shear = (0., 0.)
    shift = 0.
    interpolation = E.InterType.NEAREST
    border_mode = E.BorderType.CONSTANT
    fill_value = 0
    dim = C.AXIAL_DIM
    shape = (4, 4, 1, 1)

    tpoints = FG.plane_affine(
        points,
        scale,
        angle,
        shear,
        shift,
        dim,
        shape,
    )

    tmask = FV.plane_affine(
        mask,
        scale,
        angle,
        shear,
        shift,
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
        [2.5, 0.5, 0.5, 1.0],
    ])

    scale = (1., 1.)
    angle = 90.
    shear = (0., 0.)
    shift = 0.
    dim = C.AXIAL_DIM
    shape = (4, 2, 1, 1)

    output = FG.plane_affine(
        input,
        scale,
        angle,
        shear,
        shift,
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

    scale = (1., 1.)
    angle = 90.
    shear = (0., 0.)
    shift = 0.
    interpolation = E.InterType.NEAREST
    border_mode = E.BorderType.CONSTANT
    fill_value = 0
    dim = C.AXIAL_DIM
    shape = (4, 2, 1, 1)

    tpoints = FG.plane_affine(
        points,
        scale,
        angle,
        shear,
        shift,
        dim,
        shape,
    )

    tmask = FV.plane_affine(
        mask,
        scale,
        angle,
        shear,
        shift,
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

    scale = (2., 2.)
    angle = 0.
    shear = (0., 0.)
    shift = 0.
    dim = C.AXIAL_DIM
    shape = (4, 4, 1, 1)

    output = FG.plane_affine(
        input,
        scale,
        angle,
        shear,
        shift,
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

    scale = (2., 2.)
    angle = 0.
    shear = (0., 0.)
    shift = 0.
    interpolation = E.InterType.NEAREST
    border_mode = E.BorderType.CONSTANT
    fill_value = 0
    dim = C.AXIAL_DIM
    shape = (4, 4, 1, 1)

    tpoints = FG.plane_affine(
        points,
        scale,
        angle,
        shear,
        shift,
        dim,
        shape,
    )

    tmask = FV.plane_affine(
        mask,
        scale,
        angle,
        shear,
        shift,
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

    scale = (0.5, 0.5)
    angle = 0.
    shear = (0., 0.)
    shift = 0.
    interpolation = E.InterType.NEAREST
    border_mode = E.BorderType.CONSTANT
    fill_value = 0
    dim = C.AXIAL_DIM
    shape = (4, 4, 1, 1)

    tpoints = FG.plane_affine(
        points,
        scale,
        angle,
        shear,
        shift,
        dim,
        shape,
    )

    tmask = FV.plane_affine(
        mask,
        scale,
        angle,
        shear,
        shift,
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

    scale = (1., 1.)
    angle = 0
    shear = (0., 0.)
    shift = 0.25
    dim = C.AXIAL_DIM
    shape = (4, 4, 1, 1)

    output = FG.plane_affine(
        input,
        scale,
        angle,
        shear,
        shift,
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

    scale = (1., 1.)
    angle = 0
    shear = (0., 0.)
    shift = 0.25
    interpolation = E.InterType.NEAREST
    border_mode = E.BorderType.CONSTANT
    fill_value = 0
    dim = C.AXIAL_DIM
    shape = (4, 4, 1, 1)

    tpoints = FG.plane_affine(
        points,
        scale,
        angle,
        shear,
        shift,
        dim,
        shape,
    )

    tmask = FV.plane_affine(
        mask,
        scale,
        angle,
        shear,
        shift,
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
def test_plane_affine_CASE_only_anisotropic_shift():
    input = np.array([
        [ 0.5, 0.5, 0.5, 1.0],
        [ 2.5, 1.5, 0.5, 1.0],

    ])

    expected = np.array([
        [ 1.5, 2.5, 0.5, 1.0],
        [ 3.5, 3.5, 0.5, 1.0],
    ])

    scale = (1., 1.)
    angle = 0
    shear = (0., 0.)
    shift = (0.25, 0.5)
    dim = C.AXIAL_DIM
    shape = (4, 4, 1, 1)

    output = FG.plane_affine(
        input,
        scale,
        angle,
        shear,
        shift,
        dim,
        shape,
    )

    assert np.allclose(output, expected)

@pytest.mark.functional
def test_plane_affine_CASE_only_anisotropic_shift_AND_align_with_mask():
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

    scale = (1., 1.)
    angle = 0
    shear = (0., 0.)
    shift = (0.25, 0.5)
    interpolation = E.InterType.NEAREST
    border_mode = E.BorderType.CONSTANT
    fill_value = 0
    dim = C.AXIAL_DIM
    shape = (4, 4, 1, 1)

    tpoints = FG.plane_affine(
        points,
        scale,
        angle,
        shear,
        shift,
        dim,
        shape,
    )

    tmask = FV.plane_affine(
        mask,
        scale,
        angle,
        shear,
        shift,
        interpolation,
        border_mode,
        fill_value,
        dim,
    )

    for xidx, yidx, zidx in (tpoints + C.C2I_SHIFT)[:, :C.NUM_SPATIAL_COORDS].astype(np.int32):
        assert tmask[xidx, yidx, zidx] == 1

        tmask[xidx, yidx, zidx] = 0

    assert np.all(tmask == 0)
