import numpy as np

def get_rotation_matrix(angle):
    """Get 3x3 clockwise rotation matrix for planar transformation

        :args:
            angle: float
                angle in degrees
    """
    angle = np.deg2rad(angle)

    M = np.array([
        [ np.cos(angle), -np.sin(angle), 0. ],
        [ np.sin(angle),  np.cos(angle), 0. ],
        [            0.,             0., 1. ],
    ])

    return M

def get_translation_matrix(shifts):
    """Get 3x3 translation matrix for planar transformation

        :args:
            shifts: (float, float)
                shifts in pixels
    """
    xshift, yshift = shifts

    M = np.array([
        [ 1., 0., xshift ],
        [ 0., 1., yshift ],
        [ 0., 0.,     1. ],
    ])

    return M

def get_scaling_matrix(scales):
    """Get 3x3 scaling matrix for planar transformation

        :args:
            scales: (float, float)
                scales as ratio
    """
    xscale, yscale = scales

    M = np.array([
        [ xscale,     0., 0. ],
        [     0., yscale, 0. ],
        [     0.,     0., 1. ],
    ])

    return M

def get_shear_matrix(shears):
    """Get 3x3 shear matrix for planar transformation

        :args:
            shears: (float, float)
                shears in degrees
    """
    xshear, yshear = shears
    xshear = np.deg2rad(xshear)
    yshear = np.deg2rad(yshear)

    M = np.array([
        [ 1.            , np.tan(xshear), 0. ],
        [ np.tan(yshear),             1., 0. ],
        [             0.,             0., 1. ],
    ])

    return M

def get_affine_matrix(scales, shiftes, angle):
    T1 = get_scaling_matrix(scales)

    T3 = get_translation_matrix(shiftes)
    T4 = get_rotation_matrix(angle)

    return T4 @ T3 @ T1
