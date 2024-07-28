import cv2
import scipy as sc
import numpy as np

import voxelmentations.core.enum as E
import voxelmentations.core.constants as C
import voxelmentations.core.functional as F

def apply_along_dim(data, func, dim):
    """Apply the same transformation along the dim
    """
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
    """Perform clockwise rotation of planes along the dim
    """
    shape = (*voxel.shape[:dim], *voxel.shape[dim+1:C.NUM_SPATIAL_DIMENSIONS])

    point = [ 0.5 * ishape - 0.5 for ishape in shape ][::-1]
    T = cv2.getRotationMatrix2D(point, -angle, 1.)

    func = lambda arr: cv2.warpAffine(
        arr,
        T,
        shape,
        flags=C.MAP_INTER_TO_CV2[interpolation],
        borderMode=C.MAP_BORDER_TYPE_TO_CV2[border_mode],
        borderValue=fill_value,
    )

    voxel = apply_along_dim(voxel, func, dim)

    return voxel

@F.preserve_channel_dim
def plane_scale(voxel, scale, interpolation, border_mode, fill_value, dim):
    shape = (*voxel.shape[:dim], *voxel.shape[dim+1:C.NUM_SPATIAL_DIMENSIONS])

    point = [ 0.5 * ishape - 0.5 for ishape in shape ][::-1]
    T = cv2.getRotationMatrix2D(point, 0., scale)

    func = lambda arr: cv2.warpAffine(
        arr,
        T,
        shape,
        flags=C.MAP_INTER_TO_CV2[interpolation],
        borderMode=C.MAP_BORDER_TYPE_TO_CV2[border_mode],
        borderValue=fill_value,
    )

    voxel = apply_along_dim(voxel, func, dim)

    return voxel

@F.preserve_channel_dim
def plane_affine(voxel, angle, shift, scale, interpolation, border_mode, fill_value, dim):
    shape = (*voxel.shape[:dim], *voxel.shape[dim+1:C.NUM_SPATIAL_DIMENSIONS])

    point = [ 0.5 * ishape - 0.5 for ishape in shape ][::-1]
    T = cv2.getRotationMatrix2D(point, -angle, scale)
    T[:, 2] += np.array(shape) * shift

    func = lambda arr: cv2.warpAffine(
        arr,
        T,
        shape,
        flags=C.MAP_INTER_TO_CV2[interpolation],
        borderMode=C.MAP_BORDER_TYPE_TO_CV2[border_mode],
        borderValue=fill_value,
    )

    voxel = apply_along_dim(voxel, func, dim)

    return voxel

def gauss_noise(voxel, gauss):
    if len(voxel.shape) > len(gauss.shape):
        gauss = np.expand_dims(gauss, axis=-1)

    return voxel + gauss

def conv(voxel, kernel, border_mode, fill_value):
    if len(voxel.shape) == C.NUM_MULTI_CHANNEL_DIMENSIONS:
        kernel = np.expand_dims(kernel, axis=-1)

    return sc.ndimage.convolve(
        voxel,
        kernel,
        mode=C.MAP_BORDER_TYPE_TO_SC[border_mode],
        cval=fill_value,
    )

def _plane_grid_distortion_cv_engine():
    ...

def _plane_grid_distortion_sc_engine():
    ...

def plane_grid_distortion():
    ...
