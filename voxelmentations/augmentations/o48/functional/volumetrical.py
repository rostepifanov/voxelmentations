import numpy as np

import voxelmentations.augmentations.utils.decorators as D

@D.contiguous
def flip(voxel, dims):
    """Flip along dims

        :args:
            dims: tuple of int
                the value of dims to flip
    """
    return np.flip(voxel, dims)

@D.contiguous
def rot90(voxel, dims, times):
    """Rotate clockwise in plane formed by dims

        :args:
            dims: (int, int)
                the value of dims to form rotating plane
            times: int
                the number of plane rotation by 90 degrees
    """
    return np.rot90(voxel, times, dims[::-1])

@D.contiguous
def transpose(voxel, dims):
    """Transose a plane formed by dims

        :args:
            dims: (int, int)
                the value of dims to permute
    """
    return np.swapaxes(voxel, *dims)
