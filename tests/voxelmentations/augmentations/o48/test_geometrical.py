import pytest

import numpy as np

import voxelmentations.core.enum as E
import voxelmentations.core.constants as C

from voxelmentations.augmentations.o48.functional import FV, FG

FLIP = {
    'vertical_dim_CASE_inside_points': {
        'input': np.array([
            [1, 2, 3, 1],
            [3, 5, 2, 1],
        ]),
        'expected': np.array([
            [7, 2, 3, 1],
            [5, 5, 2, 1],
        ]),
        'args': {
            'dims': [0, ],
            'shape': (8, 8, 8),
        },
    },
    'axial_plane_CASE_inside_points': {
        'input': np.array([
            [1, 2, 3, 1],
            [3, 5, 2, 1],
        ]),
        'expected': np.array([
            [7, 6, 3, 1],
            [5, 3, 2, 1],
        ]),
        'args': {
            'dims': [0, 1],
            'shape': (8, 8, 8),
        },
    },
    'vertical_dim_CASE_outside_points': {
        'input': np.array([
            [-4, 2, 3, 1],
            [10, 5, 2, 1],
        ]),
        'expected': np.array([
            [12, 2, 3, 1],
            [-2, 5, 2, 1],
        ]),
        'args': {
            'dims': [0, ],
            'shape': (8, 8, 8),
        },
    },
}

FLIP_ALIGN_WITH_MASK = {
    'vertical_dim_AND_square_shape': {
        'points': np.array([
            [1.5, 0.5, 0.5, 1.0],
        ]),
        'mask': np.expand_dims(
            np.array([
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]),
            axis=(2, 3),
        ),
        'args': {
            'dims': [0, ],
        },
    },
    'vertical_dim_AND_rectangle_shape': {
        'points': np.array([
            [1.5, 0.5, 0.5, 1.0],
        ]),
        'mask': np.expand_dims(
            np.array([
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]),
            axis=(2, 3),
        ),
        'args': {
            'dims': [0, ],
        },
    },
}

ROT90 = {
    'inside_points_AND_one_times': {
        'input': np.array([
            [1.5, 0.5, 1., 1.],
            [2.5, 1.5, 1., 1.],
            [1.5, 1.5, 1., 1.],
        ]),
        'expected': np.array([
            [0.5, 2.5, 1., 1.],
            [1.5, 1.5, 1., 1.],
            [1.5, 2.5, 1., 1.],
        ]),
        'args': {
            'dims': [0, 1],
            'times': 1,
            'shape': (4, 4, 4),
        },
    },
    'outside_points_AND_one_times': {
        'input': np.array([
            [-1.5,  0.5, 1., 1.],
            [-2.5, -1.5, 1., 1.],
        ]),
        'expected': np.array([
            [0.5,  5.5, 1., 1.],
            [-1.5, 6.5, 1., 1.],
        ]),
        'args': {
            'dims': [0, 1],
            'times': 1,
            'shape': (4, 4, 4),
        },
    },
    'inside_points_AND_two_times': {
        'input': np.array([
            [1.5, 0.5, 1., 1.],
            [1.5, 1.5, 1., 1.],
        ]),
        'expected': np.array([
            [2.5, 3.5, 1., 1.],
            [2.5, 2.5, 1., 1.],
        ]),
        'args': {
            'dims': [0, 1],
            'times': 2,
            'shape': (4, 4, 4),
        },
    },
    'inside_points_AND_three_times': {
        'input': np.array([
            [1.5, 0.5, 1., 1.],
            [1.5, 1.5, 1., 1.],
        ]),
        'expected': np.array([
            [3.5, 1.5, 1., 1.],
            [2.5, 1.5, 1., 1.],
        ]),
        'args': {
            'dims': [0, 1],
            'times': 3,
            'shape': (4, 4, 4),
        },
    },
}

ROT90_ALIGN_WITH_MASK = {
    'one_times_AND_square_shape': {
        'points': np.array([
            [1.5, 0.5, 0.5, 1.0],
        ]),
        'mask': np.expand_dims(
            np.array([
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]),
            axis=(2, 3),
        ),
        'args': {
            'dims': [0, 1],
            'times': 1,
        },
    },
    'one_times_AND_rectangle_shape': {
        'points': np.array([
            [1.5, 0.5, 0.5, 1.0],
        ]),
        'mask': np.expand_dims(
            np.array([
                [ 0, 0],
                [ 1, 0],
                [ 0, 0],
                [ 0, 0],
            ]),
            axis=(2, 3),
        ),
        'args': {
            'dims': [0, 1],
            'times': 1,
        },
    },
    'two_times_AND_square_shape': {
        'points': np.array([
            [1.5, 0.5, 0.5, 1.0],
        ]),
        'mask': np.expand_dims(
            np.array([
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]),
            axis=(2, 3),
        ),
        'args': {
            'dims': [0, 1],
            'times': 2,
        },
    },
    'two_times_AND_rectangle_shape': {
        'points': np.array([
            [1.5, 0.5, 0.5, 1.0],
        ]),
        'mask': np.expand_dims(
            np.array([
                [0, 0],
                [1, 0],
                [0, 0],
                [0, 0],
            ]),
            axis=(2, 3),
        ),
        'args': {
            'dims': [0, 1],
            'times': 2,
        },
    },
    'three_times_AND_square_shape': {
        'points': np.array([
            [1.5, 0.5, 0.5, 1.0],
        ]),
        'mask': np.expand_dims(
            np.array([
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]),
            axis=(2, 3),
        ),
        'args': {
            'dims': [0, 1],
            'times': 3,
        },
    },
    'three_times_AND_rectangle_shape': {
        'points': np.array([
            [1.5, 0.5, 0.5, 1.0],
        ]),
        'mask': np.expand_dims(
            np.array([
                [0, 0],
                [1, 0],
                [0, 0],
                [0, 0],
            ]),
            axis=(2, 3),
        ),
        'args': {
            'dims': [0, 1],
            'times': 3,
        },
    },
}

TRANSPOSE = {
    'ordered_dims': {
        'input': np.array([
            [1, 2, 3, 1],
            [4, 5, 2, 1],
        ]),
        'expected': np.array([
            [2, 1, 3, 1],
            [5, 4, 2, 1],
        ]),
        'args': {
            'dims': [0, 1],
        },
    },
    'reversed_dims': {
        'input': np.array([
            [1, 2, 3, 1],
            [4, 5, 2, 1],
        ]),
        'expected': np.array([
            [2, 1, 3, 1],
            [5, 4, 2, 1],
        ]),
        'args': {
            'dims': [1, 0],
        },
    },
}

TRANSPOSE_ALIGN_WITH_MASK = {
    'axial_plane_CASE_square_shape': {
        'points': np.array([
            [1.5, 0.5, 0.5, 1.0],
        ]),
        'mask': np.expand_dims(
            np.array([
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]),
            axis=(2, 3),
        ),
        'args': {
            'dims': [0, 1],
        },
    },
    'axial_plane_CASE_rectangle_shape': {
        'points': np.array([
            [1.5, 0.5, 0.5, 1.0],
        ]),
        'mask': np.expand_dims(
            np.array([
                [0, 0],
                [1, 0],
                [0, 0],
                [0, 0],
            ]),
            axis=(2, 3),
        ),
        'args': {
            'dims': [0, 1],
        },
    },
}

@pytest.mark.functional
@pytest.mark.o48
@pytest.mark.flip
@pytest.mark.parametrize('name', FLIP.keys())
def test_flip(name):
    case = FLIP[name]

    output = FG.flip(
        case['input'],
        **case['args'],
    )

    assert np.allclose(output, case['expected'])

@pytest.mark.functional
@pytest.mark.o48
@pytest.mark.flip
@pytest.mark.parametrize('name', FLIP_ALIGN_WITH_MASK.keys())
def test_flip_AND_align_with_mask(name):
    case = FLIP_ALIGN_WITH_MASK[name]

    tmask = FV.flip(
        case['mask'],
        **case['args'],
    )

    tpoints = FG.flip(
        case['points'],
        shape=case['mask'].shape,
        **case['args'],
    )

    for xidx, yidx, zidx in (tpoints + C.C2I_SHIFT)[:, :C.NUM_SPATIAL_COORDS].astype(np.int32):
        assert tmask[xidx, yidx, zidx] == 1

        tmask[xidx, yidx, zidx] = 0

    assert np.all(tmask == 0)

@pytest.mark.functional
@pytest.mark.o48
@pytest.mark.rot90
@pytest.mark.parametrize('name', ROT90.keys())
def test_rot90(name):
    case = ROT90[name]

    output = FG.rot90(
        case['input'],
        **case['args'],
    )

    assert np.allclose(output, case['expected'])

@pytest.mark.functional
@pytest.mark.o48
@pytest.mark.rot90
def test_rot90_CASE_inside_points_AND_opposite_directions():
    input = np.array([
        [1.5, 0.5, 1., 1.],
        [2.5, 1.5, 1., 1.],
        [1.5, 1.5, 1., 1.],
    ])

    expected = np.array([
        [0.5, 2.5, 1., 1.],
        [1.5, 1.5, 1., 1.],
        [1.5, 2.5, 1., 1.],
    ])

    dims1 = [0, 1]
    times1 = 1
    shape1 = (4, 4, 4)

    output1 = FG.rot90(input, dims1, times1, shape1)

    dims2 = [1, 0]
    times2 = 3
    shape2 = (4, 4, 4)

    output2 = FG.rot90(input, dims2, times2, shape2)

    assert np.allclose(output1, output2)

@pytest.mark.functional
@pytest.mark.o48
@pytest.mark.rot90
@pytest.mark.parametrize('name', ROT90_ALIGN_WITH_MASK.keys())
def test_rot90_AND_align_with_mask(name):
    case = ROT90_ALIGN_WITH_MASK[name]

    tmask = FV.rot90(
        case['mask'],
        **case['args'],
    )

    tpoints = FG.rot90(
        case['points'],
        shape=case['mask'].shape,
        **case['args'],
    )

    for xidx, yidx, zidx in (tpoints + C.C2I_SHIFT)[:, :C.NUM_SPATIAL_COORDS].astype(np.int32):
        assert tmask[xidx, yidx, zidx] == 1

        tmask[xidx, yidx, zidx] = 0

    assert np.all(tmask == 0)

@pytest.mark.functional
@pytest.mark.o48
@pytest.mark.transpose
@pytest.mark.parametrize('name', TRANSPOSE.keys())
def test_transpose(name):
    case = TRANSPOSE[name]

    output = FG.transpose(
        case['input'],
        **case['args'],
    )

    assert np.allclose(output, case['expected'])

@pytest.mark.functional
@pytest.mark.o48
@pytest.mark.transpose
@pytest.mark.parametrize('name', TRANSPOSE_ALIGN_WITH_MASK.keys())
def test_rot90_AND_align_with_mask(name):
    case = TRANSPOSE_ALIGN_WITH_MASK[name]

    tmask = FV.transpose(
        case['mask'],
        **case['args'],
    )

    tpoints = FG.transpose(
        case['points'],
        **case['args'],
    )

    for xidx, yidx, zidx in (tpoints + C.C2I_SHIFT)[:, :C.NUM_SPATIAL_COORDS].astype(np.int32):
        assert tmask[xidx, yidx, zidx] == 1

        tmask[xidx, yidx, zidx] = 0

    assert np.all(tmask == 0)
