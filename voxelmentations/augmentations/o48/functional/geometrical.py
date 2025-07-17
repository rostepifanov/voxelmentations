import numpy as np

def flip(points, dims, shape):
    """Flip along dims
    """
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
        points[:, dims[1]] = np.subtract(shape[dims[0]], points[:, dims[1]])
    elif times % 4 == 2:
        points[:, dims] = np.subtract(shape[[*dims]], points[:, dims])
    else:
        points[:, dims] = points[:, dims[::-1]]
        points[:, dims[0]] = np.subtract(shape[dims[1]], points[:, dims[0]])

    return points

def transpose(points, dims):
    """Transpose a plane formed by dims
    """
    points = np.copy(points)

    points[:, dims] = points[:, dims[::-1]]

    return points