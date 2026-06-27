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
    def at(ndim, idx):
        index = [slice(None)] * ndim
        index[dim] = idx
        return tuple(index)

    dim = dim % data.ndim

    first = func(data[at(data.ndim, 0)])

    shape = (*first.shape[:dim], data.shape[dim], *first.shape[dim:])
    out = np.empty(shape, dtype=first.dtype)

    out[at(out.ndim, 0)] = first

    for idx in range(1, data.shape[dim]):
        out[at(out.ndim, idx)] = func(
            data[at(data.ndim, idx)]
        )

    return out

def pad(voxel, pads, border_mode, fill_value):
    """Pad voxel

        :args:
            pads: tuple of (int, int)
                the size of padding for each dims
            border_mode: BorderType
                border mode
            fill_value:
                padding value if border_mode is BorderType.CONSTANT
    """
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

def affine(voxel, scale, angles, shift, interpolation, border_mode, fill_value):
    """Apply scale transformations to volume

        :NOTE:
            Scipy uses "backward" notation that means I = M @ O + offset,
            where I is input image, O is output image.

            Ordinary it is used "forward" notation that means I and O are swapped:
                O = M' @ I + offset'
                I = inv(M') @ O - inv(M') @ offset'

            Thus, M = inv(M') and offset = inv(M') @ offset',
            where M and offset are args of affine_transform.

            Matrix M encodes the scaling and rotation,
            while the vector offset encodes the translation.

            Dimension beyond C.NUM_SPATIAL_DIMENSIONS
            should be modified using the identity transformation.

            In the case where the number of dimensions is equal to C.NUM_MULTI_CHANNEL_DIMENSIONS,
            the value 1 in the 4×4 matrix S represents the identity transformation,
            not a homogeneous coordinate — despite the similar notation.

        :args:
            scale: (float, float, float)
                scaling factor in range from 0 to 1
            angles: (float, float, float)
                angles to rotate in range from 0 to 180
            shift: (float, float, float)
                translation factor in range from 0 to 1
            interpolation: InterType
                interpolation mode
            border_mode: BorderType
                border mode
            fill_value:
                padding value if border_mode is BorderType.CONSTANT
    """
    shape = voxel.shape[:C.NUM_SPATIAL_DIMENSIONS]
    shape = np.array(shape)

    shift = (shape * shift)
    point = (shape - 1) / 2

    S = G.get_volumetric_scaling_matrix(scale)
    R = G.get_volumetric_rotation_matrix(angles)

    if len(voxel.shape) == C.NUM_MULTI_CHANNEL_DIMENSIONS:
        point = np.array([*point, 0])
        shift = np.array([*shift, 0])
    else:
        S = S[:C.NUM_SPATIAL_DIMENSIONS, :C.NUM_SPATIAL_DIMENSIONS]
        R = R[:C.NUM_SPATIAL_DIMENSIONS, :C.NUM_SPATIAL_DIMENSIONS]

    dS = S.diagonal()
    invS = np.diag(1 / dS)

    invR = R.T

    offset = point - invR @ (point + shift) / dS

    voxel = sc.ndimage.affine_transform(
        voxel,
        invR @ invS,
        offset=offset,
        order=C.MAP_INTER_TO_SC[interpolation],
        mode=C.MAP_BORDER_TYPE_TO_SC[border_mode],
        cval=fill_value,
    )

    return voxel

@D.preserve_channel_dim
def plane_affine(voxel, scale, angle, shear, shift, interpolation, border_mode, fill_value, dim):
    """Apply affine transformations to plane orthogonal to the dim

        :NOTE:
            Since the plane is rotated relative to its center, the matematical notation of transformation is:
                K @ T @ inv(K),
            where K is the translation of the origin to the center of plane.

            Some confusions of plane center in OpenCV:
                1. https://answers.opencv.org/question/182793/centering-opencv-rotation/
                2. https://answers.opencv.org/question/35111/origin-pixel-in-the-image-coordinate-system-in-opencv/

            Since the right lower corner has coordinates equal to (height - 1, width - 1),
            than plane center in OpenCV has coordinates equal to (height - 1) / 2, (width - 1) / 2.

        :args:
            scale: (float, float)
                scaling factor in range from 0 to 1
            angle: float
                angle of rotation in range from 0 to 180
            shear: (float, float)
                shear factor in range from 0 to 180
            shift: (float, float)
                translation factor in range from 0 to 1
            interpolation: InterType
                interpolation mode
            border_mode: BorderType
                border mode
            fill_value:
                padding value if border_mode is BorderType.CONSTANT
            dim:
                the plane orthogonal to the dim is transformated
    """
    shape = [*voxel.shape[:dim], *voxel.shape[dim+1:C.NUM_SPATIAL_DIMENSIONS]][::-1]
    shape = np.array(shape)

    scale = scale[::-1]
    shift = (shape * shift)[::-1]
    point = (shape - 1) / 2

    K = G.get_planar_translation_matrix(point)
    invK = G.get_planar_translation_matrix(-point)

    T = G.get_planar_affine_matrix(scale, -angle, shear, shift)

    M = K @ T @ invK
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

    return np.add(voxel, other, out=voxel)

def multiply(voxel, factor):
    return np.multiply(voxel, factor, out=voxel)

def conv(voxel, kernel, border_mode, fill_value):
    if len(voxel.shape) == C.NUM_MULTI_CHANNEL_DIMENSIONS:
        kernel = np.expand_dims(kernel, axis=C.CHANNEL_DIM)

    return sc.ndimage.convolve(
        voxel,
        kernel,
        mode=C.MAP_BORDER_TYPE_TO_SC[border_mode],
        cval=fill_value,
    )

def gaussian_blur(voxel, variance, kernel_size, border_mode, fill_value):
    sigma = variance ** 0.5

    if sigma == 0:
        return voxel

    truncate = (kernel_size - 1) / (2 * sigma)

    if len(voxel.shape) == C.NUM_MULTI_CHANNEL_DIMENSIONS:
        sigma = (sigma, sigma, sigma, 0)
    else:
        sigma = (sigma, sigma, sigma)

    return sc.ndimage.gaussian_filter(
        voxel,
        sigma=sigma,
        mode=C.MAP_BORDER_TYPE_TO_SC[border_mode],
        cval=fill_value,
        truncate=truncate,
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

def contrast(voxel, contrast):
    """
        :NOTE:
            mathematical notation:
                voxel = (voxel - mean) * contrast + mean
    """
    mean = np.mean(voxel)

    np.subtract(voxel, mean, out=voxel)
    np.multiply(voxel, contrast, out=voxel)

    return np.add(voxel, mean, out=voxel)

def gamma(voxel, gamma):
    """
        :NOTE:
            mathematical notation:
                voxel = signs * voxel^gamma
    """
    signs = np.sign(voxel)
    np.abs(voxel, out=voxel)
    np.power(voxel, gamma, out=voxel)

    return np.multiply(signs, voxel, out=voxel)

def plane_dropout(voxel, indices, value, dim):
    selector = [ slice(0, None) for _ in range(dim) ]
    voxel[(*selector, indices)] = value

    return voxel

def patch_dropout(voxel, patches, value):
    for patch in patches:
        voxel[patch] = value

    return voxel

def patch_shuffle(voxel, patches):
    for patch in patches:
        pixels = voxel[patch]
        shape = pixels.shape

        if pixels.ndim == C.NUM_MULTI_CHANNEL_DIMENSIONS:
            flat = pixels.reshape(-1, shape[C.CHANNEL_DIM])
        else:
            flat = pixels.reshape(-1)

        np.random.shuffle(flat)

        if not np.shares_memory(flat, pixels):
            voxel[patch] = flat.reshape(shape)

    return voxel

def rescale(voxel, scale, interpolation):
    voxel = apply_along_dim(
        voxel,
        lambda arr: sc.ndimage.zoom(
            arr,
            scale,
            order=C.MAP_INTER_TO_SC[interpolation],
            mode='grid-constant',
            grid_mode=True,
        ),
        dim=-1,
    )

    return voxel

@D.preserve_channel_dim
def reshape(voxel, shape, interpolation):
    voxel = apply_along_dim(
        voxel,
        lambda arr: cv2.resize(
            arr,
            (shape[1], shape[0]),
            fx=None,
            fy=None,
            interpolation=C.MAP_INTER_TO_CV2[interpolation],
        ),
        dim=2,
    )

    voxel = apply_along_dim(
        voxel,
        lambda arr: cv2.resize(
            arr,
            (shape[2], voxel.shape[1]),
            fx=None,
            fy=None,
            interpolation=C.MAP_INTER_TO_CV2[interpolation],
        ),
        dim=0,
    )

    return voxel

def downscale(voxel, scale, down_interpolation, up_interpolation):
    shape = voxel.shape[:C.NUM_SPATIAL_DIMENSIONS]

    voxel = rescale(voxel, scale, down_interpolation)
    voxel = reshape(voxel, shape, up_interpolation)

    return voxel
