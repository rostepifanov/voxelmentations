import cv2

import voxelmentations.core.enum as E

VERTICAL_DIM = 0
HORIZONTAL_DIM = 1
AXIAL_DIM = 2
CHANNEL_DIM = 3

SPATIAL_DIMS = (0, 1, 2)

NUM_SPATIAL_DIMENSIONS = 3
NUM_MONO_CHANNEL_DIMENSIONS = 3
NUM_MULTI_CHANNEL_DIMENSIONS = 4

MAP_BORDER_TYPE_TO_CV2 = {
    E.BorderType.CONSTANT: cv2.BORDER_CONSTANT,
    E.BorderType.REPLICATE: cv2.BORDER_REPLICATE,
    E.BorderType.REFLECT_1001: cv2.BORDER_REFLECT,
    E.BorderType.REFLECT_101: cv2.BORDER_REFLECT_101,
    E.BorderType.WRAP: cv2.BORDER_WRAP,
}

MAP_BORDER_TYPE_TO_NUMPY = {
    E.BorderType.CONSTANT: 'constant',
    E.BorderType.REPLICATE: 'edge',
    E.BorderType.REFLECT_1001: 'symmetric',
    E.BorderType.REFLECT_101: 'reflect',
    E.BorderType.WRAP: 'wrap',
}

MAP_BORDER_TYPE_TO_SC = {
    E.BorderType.CONSTANT: 'constant',
    E.BorderType.REPLICATE: 'nearest',
    E.BorderType.REFLECT_1001: 'reflect',
    E.BorderType.REFLECT_101: 'mirror',
    E.BorderType.WRAP: 'wrap',
}

MAP_INTER_TO_CV2 = {
    E.InterType.NEAREST: cv2.INTER_NEAREST,
    E.InterType.LINEAR: cv2.INTER_LINEAR,
}

MAP_INTER_TO_SC = {
    E.InterType.NEAREST: 0,
    E.InterType.LINEAR: 1,
}
