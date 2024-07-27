import pytest

import cv2
import numpy as np

import voxelmentations.core.enum as E
import voxelmentations.core.constants as C
import voxelmentations.augmentations.functional as F

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

    output = F.plane_dropout(input, indices, fill_value, dim)

    assert np.allclose(output, expected)

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

    output = F.pad(input, pads, border_mode, fill_value)

    assert np.allclose(output, expected)

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

    output = F.pad(input, pads, border_mode, fill_value)

    assert np.allclose(output, expected)

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

    output = F.pad(input, pads, border_mode, fill_value)

    assert np.allclose(output, expected)

def test_plane_rotate_CASE_90_degree():
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

    angle = 90
    interpolation = E.InterType.NEAREST
    border_mode = E.BorderType.CONSTANT
    fill_value = 0
    dim = C.AXIAL_DIM

    output = F.plane_rotate(
        input,
        angle,
        interpolation,
        border_mode,
        fill_value,
        dim
    )

    assert np.allclose(output, expected)

def test_plane_scale_CASE_upscale_twice():
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

    scale = 2
    interpolation = E.InterType.NEAREST
    border_mode = E.BorderType.CONSTANT
    fill_value = 0
    dim = C.AXIAL_DIM

    output = F.plane_scale(
        input,
        scale,
        interpolation,
        border_mode,
        fill_value,
        dim
    )

    assert np.allclose(output, expected)

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

    output = F.plane_affine(
        input,
        angle,
        shift,
        scale,
        interpolation,
        border_mode,
        fill_value,
        dim
    )

    assert np.allclose(output, expected)

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
            [ 0,  0,  0,  0],
            [ 0, 13,  9,  5],
            [ 0, 14, 10,  6],
            [ 0, 15, 11 , 7],
        ]),
        axis=(2, 3)
    )

    angle = 90
    shift = 0.25
    scale = 1
    interpolation = E.InterType.NEAREST
    border_mode = E.BorderType.CONSTANT
    fill_value = 0
    dim = C.AXIAL_DIM

    output = F.plane_affine(
        input,
        angle,
        shift,
        scale,
        interpolation,
        border_mode,
        fill_value,
        dim
    )

    assert np.allclose(output, expected)
