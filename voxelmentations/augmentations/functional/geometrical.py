import numpy as np

def transpose(points, dims):
    points = np.copy(points)
    points[:, dims] = points[:, dims[::-1]]

    return points
