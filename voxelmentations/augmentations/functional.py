import cv2
import numpy as np

import voxelmentations.core.enum as E
import voxelmentations.core.constants as C
import voxelmentations.core.functional as F

def apply_along_dim(data, func, dim):
    data = np.moveaxis(data, dim, 0)

    data = np.stack([*map(
        func,
        data,
    )], axis=dim)

    return data

def pad(voxel, pads, border_mode, fill_value):
    kwargs = dict()

    if len(voxel.shape) == C.NUM_MULTI_CHANNEL_DIMENSIONS:
        pads = (*pads, (0, 0))

    if border_mode == E.BorderType.CONSTANT:
        kwargs['constant_values'] = fill_value

    voxel = np.pad( voxel,
                    pad_width=pads,
                    mode=C.MAP_BORDER_TYPE_TO_NUMPY[border_mode],
                    **kwargs )

    return voxel

def flip(voxel, dims):
    return np.flip(voxel, dims)

def plane_dropout(voxel, indices, value, dim):
    voxel = np.copy(voxel)

    selector = [ slice(0, None) for _ in range(dim) ]
    voxel[(*selector, indices)] = value

    return voxel

@F.preserve_channel_dim
def plane_rotate(voxel, angle, interpolation, border_mode, fill_value, dim):
    shape = (*voxel.shape[:dim], *voxel.shape[dim+1:C.NUM_SPATIAL_DIMENSIONS])

    point = [ 0.5 * ishape for ishape in shape ][::-1]
    T = cv2.getRotationMatrix2D(point, angle, 1.)

    func = lambda arr: cv2.warpAffine(
        arr,
        T,
        shape,
        flags=interpolation,
        borderMode=C.MAP_BORDER_TYPE_TO_CV2[border_mode],
        borderValue=fill_value,
    )

    voxel = apply_along_dim(voxel, func, dim)

    return voxel

@F.preserve_channel_dim
def plane_scale(voxel, scale, interpolation, border_mode, fill_value, dim):
    shape = (*voxel.shape[:dim], *voxel.shape[dim+1:C.NUM_SPATIAL_DIMENSIONS])

    T = np.eye(2, 3, dtype=np.float32) * scale

    func = lambda arr: cv2.warpAffine(
        arr,
        T,
        shape,
        flags=interpolation,
        borderMode=C.MAP_BORDER_TYPE_TO_CV2[border_mode],
        borderValue=fill_value,
    )

    voxel = apply_along_dim(voxel, func, dim)

    return voxel
