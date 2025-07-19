import numpy as np

def flip(points, dims, shape):
    """Flip along dims

        :args:
            dims: tuple of int
                the value of dims to flip
            shape: tuple of float
                voxel shape where the point is located
    """
    points = np.copy(points)

    shape = np.array(shape)
    points[:, dims] = np.subtract(shape[dims], points[:, dims])

    return points

def rot90(points, dims, times, shape):
    """Rotate clockwise in plane formed by dims

        :args:
            dims: (int, int)
                the value of dims to form rotating plane
            times: int
                the number of plane rotation by 90 degrees
            shape: tuple of float
                voxel shape where the point is located
    """
    points = np.copy(points)

    shape = np.array(shape)

    if times % 4 == 0:
        points = points
    elif times % 4 == 1:
        points[:, dims] = points[:, dims[::-1]]
        points[:, dims[1]] = np.subtract(shape[dims[0]], points[:, dims[1]])
    elif times % 4 == 2:
        points[:, dims] = np.subtract(shape[[*dims]], points[:, dims])
    else:
        points[:, dims] = points[:, dims[::-1]]
        points[:, dims[0]] = np.subtract(shape[dims[1]], points[:, dims[0]])

    return points

def transpose(points, dims):
    """Transpose a plane formed by dims

        :args:
            dims: (int, int)
                the value of dims to permute
    """
    points = np.copy(points)

    points[:, dims] = points[:, dims[::-1]]

    return points