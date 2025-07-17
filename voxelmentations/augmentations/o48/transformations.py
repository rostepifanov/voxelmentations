import numpy as np

import voxelmentations.core.constants as C
import voxelmentations.augmentations.checkers as M

from voxelmentations.augmentations.o48.functional import FV, FG
from voxelmentations.core import register_as_serializable, DualAugmentation

@register_as_serializable
class Flip(DualAugmentation):
    """Flip a voxel along a dim.
    """
    _DIMS = (C.HORIZONTAL_DIM, C.VERTICAL_DIM, C.AXIAL_DIM)

    def get_augmentation_init_args_names(self):
        return tuple()

    def get_params(self):
        code = np.random.randint(1, 2**len(self._DIMS))

        dims = []

        for dim in self._DIMS:
            if code % 2 == 1:
                dims = [*dims, dim]

            code = code / 2

        return {'dims': dims}

    @property
    def targets_as_params(self):
        return ['voxel']

    def get_params_dependent_on_targets(self, params):
        shape = params['voxel'].shape[:C.NUM_SPATIAL_DIMENSIONS]

        return {'shape': shape}

    def apply(self, voxel, dims, **params):
        return FV.flip(voxel, dims)

    def apply_to_points(self, points, dims, shape, **params):
        return FG.flip(points, dims, shape)

@register_as_serializable
class AxialFlip(Flip):
    """Flip a voxel in z dim.
    """
    _DIMS = (C.AXIAL_DIM, )

@register_as_serializable
class AxialPlaneFlip(Flip):
    """Flip a voxel in x-y plane.
    """
    _DIMS = (C.HORIZONTAL_DIM, C.VERTICAL_DIM)

@register_as_serializable
class Rotate90(DualAugmentation):
    """Rotate clockwise on 90 degrees by x times a voxel on orthogonal plane to a dim.
    """
    _DIMS = (C.HORIZONTAL_DIM, C.VERTICAL_DIM, C.AXIAL_DIM)

    def get_augmentation_init_args_names(self):
        return tuple()

    def get_params(self):
        code = np.random.choice(self._DIMS)
        dims = [*C.SPATIAL_DIMS[:code], *C.SPATIAL_DIMS[code+1:]]

        times = np.random.randint(1, 4)

        return {'dims': dims, 'times': times}

    @property
    def targets_as_params(self):
        return ['voxel']

    def get_params_dependent_on_targets(self, params):
        shape = params['voxel'].shape[:C.NUM_SPATIAL_DIMENSIONS]

        return {'shape': shape}

    def apply(self, voxel, dims, times, **params):
        return FV.rot90(voxel, dims, times)

    def apply_to_points(self, points, dims, times, shape, **params):
        return FG.rot90(points, dims, times, shape)

@register_as_serializable
class AxialPlaneRotate90(Rotate90):
    """Rotate on 90 degrees by x times a voxel in x-y plane.
    """
    _DIMS = (C.AXIAL_DIM, )

@register_as_serializable
class Tranpose(DualAugmentation):
    """Transpose a voxel on orthogonal plane to a dim.
    """
    _DIMS = (C.HORIZONTAL_DIM, C.VERTICAL_DIM, C.AXIAL_DIM)

    def get_augmentation_init_args_names(self):
        return tuple()

    def get_params(self):
        code = np.random.choice(self._DIMS)
        dims = (*C.SPATIAL_DIMS[:code], *C.SPATIAL_DIMS[code+1:])

        return {'dims': dims}

    def apply(self, voxel, dims, **params):
        return FV.transpose(voxel, dims)

    def apply_to_points(self, points, dims, **params):
        return FG.transpose(points, dims)

@register_as_serializable
class AxialPlaneTranpose(Tranpose):
    """Transpose a voxel in x-y plane.
    """
    _DIMS = (C.AXIAL_DIM, )
