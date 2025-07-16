import numpy as np

import voxelmentations.core.constants as C
import voxelmentations.augmentations.utils.geometrical as G

def pad(points, pads):
    pads = (*pads, (0, 0))
    points = points + np.array([x for x, _ in pads])

    return points

def flip(points, dims, shape):
    points = np.copy(points)

    shape = np.array(shape)
    points[:, dims] = np.subtract(shape[dims], points[:, dims])

    return points

def rot90(points, dims, times, shape):
    """Rotate clockwise in plane formed by dims
    """
    points = np.copy(points)

    shape = np.array(shape)

    if times % 4 == 0:
        points = points
    elif times % 4 == 1:
        points[:, dims] = points[:, dims[::-1]]
        points[:, dims[1]] = np.subtract(shape[dims[1]], points[:, dims[1]])
    elif times % 4 == 2:
        points[:, dims] = np.subtract(shape[[*dims]], points[:, dims])
    else:
        points[:, dims] = points[:, dims[::-1]]
        points[:, dims[0]] = np.subtract(shape[dims[0]], points[:, dims[0]])

    return points

def transpose(points, dims):
    points = np.copy(points)
    points[:, dims] = points[:, dims[::-1]]

    return points

def affine(points, scale, angles, shift, shape):
    points = np.copy(points)

    shape = np.array(shape)

    shift = shift * shape
    point = 0.5 * shape

    K = G.get_volumetric_translation_matrix(point)
    T = G.get_volumetric_affine_matrix(scale, angles, (0, 0, 0), shift)

    M = K @ T @ np.linalg.inv(K)

    np.matmul(points, M.T, out=points)

    return points

def plane_affine(points, scale, angle, shear, shift, dim, shape):
    points = np.copy(points)

    shape = [*shape[:dim], *shape[dim+1:C.NUM_SPATIAL_DIMENSIONS]]
    shape = np.array(shape)

    shift = shift * shape
    point = 0.5 * shape

    K = G.get_planar_translation_matrix(point)
    T = G.get_planar_affine_matrix(scale, angle, shear, shift)

    M = K @ T @ np.linalg.inv(K)

    dims = np.concatenate([
        np.arange(dim),
        np.arange(dim+1, C.NUM_COORDS)
    ])

    points[:, dims] = points[:, dims] @ M.T

    return points
