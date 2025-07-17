import numpy as np

def flip(voxel, dims):
    """Flip along dims

        :args:
            dims: tuple of int
                the value of dims to flip
    """
    voxel = np.flip(voxel, dims)

    return np.require(voxel, requirements=['C_CONTIGUOUS'])

def rot90(voxel, dims, times):
    """Rotate clockwise in plane formed by dims

        :args:
            dims: (int, int)
                the value of dims to form rotating plane
            times: int
                the number of plane rotation by 90 degrees
    """
    voxel = np.rot90(voxel, times, dims[::-1])

    return np.require(voxel, requirements=['C_CONTIGUOUS'])

def transpose(voxel, dims):
    """Transose a plane formed by dims

        :args:
            dims: (int, int)
                the value of dims to permute
    """
    voxel = np.swapaxes(voxel, *dims)

    return np.require(voxel, requirements=['C_CONTIGUOUS'])