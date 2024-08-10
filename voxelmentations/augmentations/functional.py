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

def addition(voxel, extra):
    if len(voxel.shape) > len(extra.shape):
        extra = np.expand_dims(extra, axis=C.CHANNEL_DIM)

    return voxel + extra

def conv(voxel, kernel, border_mode, fill_value):
    if len(voxel.shape) == C.NUM_MULTI_CHANNEL_DIMENSIONS:
        kernel = np.expand_dims(kernel, axis=C.CHANNEL_DIM)

    return sc.ndimage.convolve(
        voxel,
        kernel,
        mode=C.MAP_BORDER_TYPE_TO_SC[border_mode],
        cval=fill_value,
    )

def distort(voxel, distorted_grid, interpolation):
    """
        :NOTE:
            see http://pythology.blogspot.sg/2014/03/interpolation-on-regular-distorted-grid.html
    """
    shape = voxel.shape[:C.NUM_SPATIAL_DIMENSIONS]
    distorted_points = np.vstack([ points.flatten() for points in distorted_grid ])

    func = lambda arr: sc.ndimage.map_coordinates(
        arr,
        distorted_points,
        order=C.MAP_INTER_TO_SC[interpolation]
    ).reshape(*shape)

    if len(voxel.shape) == C.NUM_MULTI_CHANNEL_DIMENSIONS:
        voxel = apply_along_dim(voxel, func, C.CHANNEL_DIM)
    else:
        voxel = func(voxel)

    return voxel

def gamma_transform(voxel, gamma):
    values = np.abs(voxel)
    signs = np.sign(voxel)

    voxel = np.power(values, gamma) * signs

    return voxel

def plane_dropout(voxel, indices, value, dim):
    voxel = np.copy(voxel)

    selector = [ slice(0, None) for _ in range(dim) ]
    voxel[(*selector, indices)] = value

    return voxel

def patch_dropout(voxel, patches, value):
    voxel = np.copy(voxel)

    for patch in patches:
        voxel[patch] = value

    return voxel

def patch_shuffle(voxel, patches):
    voxel = np.copy(voxel)

    for patch in patches:
        pixels = voxel[patch]
        np.random.shuffle(pixels)

        voxel[patch] = pixels

    return voxel
