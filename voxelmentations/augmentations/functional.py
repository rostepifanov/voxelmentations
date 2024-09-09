import cv2
import scipy as sc
import numpy as np

import voxelmentations.core.enum as E
import voxelmentations.core.constants as C
import voxelmentations.augmentations.utils.decorators as D
import voxelmentations.augmentations.utils.geometrical as G

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

def rot90(voxel, dims, times):
    return np.rot90(voxel, times, dims)

def transpose(voxel, dims):
    return np.swapaxes(voxel, *dims)

@D.preserve_channel_dim
def plane_affine(voxel, scale, shift, angle, interpolation, border_mode, fill_value, dim):
    shape = [*voxel.shape[:dim], *voxel.shape[dim+1:C.NUM_SPATIAL_DIMENSIONS]][::-1]
    point = [ 0.5 * ishape - 0.5 for ishape in shape ]

    K = G.get_translation_matrix(np.array(point))
    T = G.get_affine_matrix((scale, scale), np.array(shape) * shift, angle)

    M = K @ T @ np.linalg.inv(K)
    M = M[:2]

    func = lambda arr: cv2.warpAffine(
        arr,
        M,
        shape,
        flags=C.MAP_INTER_TO_CV2[interpolation],
        borderMode=C.MAP_BORDER_TYPE_TO_CV2[border_mode],
        borderValue=fill_value,
    )

    voxel = apply_along_dim(voxel, func, dim)

    return voxel

def add(voxel, other):
    if len(voxel.shape) > len(other.shape):
        other = np.expand_dims(other, axis=C.CHANNEL_DIM)

    return voxel + other

def multiply(voxel, coef):
    return coef * voxel

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
