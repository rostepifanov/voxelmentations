import numpy as np

def flip(points, dims, shape):
    points = np.copy(points)

    shape = np.array(shape)
    points[:, dims] = np.subtract(shape[dims], points[:, dims])

    return points

def transpose(points, dims):
    points = np.copy(points)
    points[:, dims] = points[:, dims[::-1]]

    return points
