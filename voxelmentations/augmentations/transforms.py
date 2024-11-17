import numpy as np

import voxelmentations.core.enum as E
import voxelmentations.core.constants as C
import voxelmentations.augmentations.checkers as M

from voxelmentations.augmentations.functional import FV, FG
from voxelmentations.core.transforms import VoxelOnlyTransform, DualTransform, TripleTransform

class PadIfNeeded(DualTransform):
    """Pad shape of a voxel to a minimal shape.
    """
    def __init__(
            self,
            min_height=32,
            min_width=32,
            min_depth=32,
            position=E.PositionType.CENTER,
            border_mode=E.BorderType.DEFAULT,
            fill_value=0.,
            mask_fill_value=0,
            always_apply=False,
            p=1.0,
        ):
        """
            :args:
                min_height: int
                    minimal height to fill with padding
                min_width: int
                    minimal width to fill with padding
                min_depth: int
                    minimal depth to fill with padding
                position: PositionType or str
                    position of voxel
                border_mode: BorderType
                    border mode
                fill_value: int or float or None
                    padding value if border_mode is BorderType.CONSTANT
                mask_fill_value: int or None
                    padding value for mask if border_mode is BorderType.CONSTANT
        """
        super(PadIfNeeded, self).__init__(always_apply, p)

        self.min_height = M.prepare_non_negative_int(min_height, 'min_height')
        self.min_width = M.prepare_non_negative_int(min_width, 'min_width')
        self.min_depth = M.prepare_non_negative_int(min_depth, 'min_depth')

        self.position = E.PositionType(position)

        self.border_mode = border_mode
        self.fill_value = M.prepare_float(fill_value, 'fill_value')
        self.mask_fill_value = M.prepare_int(mask_fill_value, 'mask_fill_value')

    def get_transform_init_args_names(self):
        return (
            'min_height',
            'min_width',
            'min_depth',
            'position',
            'border_mode',
            'fill_value',
            'mask_fill_value'
        )

    @property
    def targets_as_params(self):
        return ['voxel']

    def get_params_dependent_on_targets(self, params):
        height, width, depth = params['voxel'].shape[:C.NUM_SPATIAL_DIMENSIONS]

        pad_height = max(0, self.min_height - height)
        pad_width = max(0, self.min_width - width)
        pad_depth = max(0, self.min_depth - depth)

        if self.position == E.PositionType.LEFT:
            left_pad = 0
            right_pad = pad_height

            top_pad = 0
            bottom_pad = pad_width

            front_pad = 0
            back_pad = pad_depth
        elif self.position == E.PositionType.CENTER:
            left_pad = pad_height // 2
            right_pad = pad_height - left_pad

            top_pad = pad_width // 2
            bottom_pad = pad_width - top_pad

            front_pad = pad_depth // 2
            back_pad = pad_depth - front_pad
        elif self.position == E.PositionType.RIGHT:
            left_pad = pad_height
            right_pad = 0

            top_pad = pad_width
            bottom_pad = 0

            front_pad = pad_depth
            back_pad = 0
        else:
            left_pad = np.random.randint(0, pad_height + 1)
            right_pad = pad_height - left_pad

            top_pad = np.random.randint(0, pad_width + 1)
            bottom_pad = pad_height - top_pad

            front_pad = np.random.randint(0, pad_depth + 1)
            back_pad = pad_height - front_pad

        pads = (
            (left_pad, right_pad),
            (top_pad, bottom_pad),
            (front_pad, back_pad)
        )

        return {'pads': pads}

    def apply(self, voxel, pads, **params):
        return FV.pad(voxel, pads, self.border_mode, self.fill_value)

    def apply_to_mask(self, mask, pads, **params):
        return FV.pad(mask, pads, self.border_mode, self.mask_fill_value)

class Flip(TripleTransform):
    """Flip a voxel along a dim.
    """
    _DIMS = (C.HORIZONTAL_DIM, C.VERTICAL_DIM, C.AXIAL_DIM)

    def get_transform_init_args_names(self):
        return tuple()

    def get_params(self):
        code = np.random.randint(1, 2**len(self._DIMS))

        dims = tuple()

        for dim in self._DIMS:
            if code % 2 == 1:
                dims = (*dims, dim)

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

    def apply_to_points(self, voxel, dims, shape, **params):
        return FG.flip(voxel, dims, shape)

class AxialFlip(Flip):
    """Flip a voxel in z dim.
    """
    _DIMS = (C.AXIAL_DIM, )

class AxialPlaneFlip(Flip):
    """Flip a voxel in x-y plane.
    """
    _DIMS = (C.HORIZONTAL_DIM, C.VERTICAL_DIM)

class Rotate90(DualTransform):
    """Rotate on 90 degrees by x times a voxel on orthogonal plane to a dim.
    """
    _DIMS = (C.HORIZONTAL_DIM, C.VERTICAL_DIM, C.AXIAL_DIM)

    def get_transform_init_args_names(self):
        return tuple()

    def get_params(self):
        code = np.random.choice(self._DIMS)
        dims = (*C.SPATIAL_DIMS[:code], *C.SPATIAL_DIMS[code+1:])

        times = np.random.randint(1, 4)

        return {'dims': dims, 'times': times}

    def apply(self, voxel, dims, times, **params):
        return FV.rot90(voxel, dims, times)

class AxialPlaneRotate90(Rotate90):
    """Rotate on 90 degrees by x times a voxel in x-y plane.
    """
    _DIMS = (C.AXIAL_DIM, )

class Tranpose(TripleTransform):
    """Transpose a voxel on orthogonal plane to a dim.
    """
    _DIMS = (C.HORIZONTAL_DIM, C.VERTICAL_DIM, C.AXIAL_DIM)

    def get_transform_init_args_names(self):
        return tuple()

    def get_params(self):
        code = np.random.choice(self._DIMS)
        dims = (*C.SPATIAL_DIMS[:code], *C.SPATIAL_DIMS[code+1:])

        return {'dims': dims}

    def apply(self, voxel, dims, **params):
        return FV.transpose(voxel, dims)

    def apply_to_points(self, points, dims, **params):
        return FG.transpose(points, dims)

class AxialPlaneTranpose(Tranpose):
    """Transpose a voxel in x-y plane.
    """
    _DIMS = (C.AXIAL_DIM, )

class AxialPlaneAffine(TripleTransform):
    """Randomly deform axial planes of a voxel.
    """
    def __init__(
            self,
            scale_limit=0.05,
            shift_limit=0.05,
            angle_limit=10,
            border_mode=E.BorderType.DEFAULT,
            interpolation=E.InterType.DEFAULT,
            fill_value=0,
            mask_fill_value=0,
            always_apply=False,
            p=0.5,
        ):
        """
            :args:
                scale_limit: float
                    limit of scaling
                shift_limit: float
                    limit of translation as ratio of size
                angle_limit: float
                    limit of rotation in degrees [0, 180]
                border_mode: BorderType
                    border mode
                interpolation: InterType
                    interpolation mode
                fill_value: float
                    padding value of voxel if border_mode is cv2.BORDER_CONSTANT
                mask_fill_value: int or None
                    padding value if border_mode is cv2.BORDER_CONSTANT. if value is None, mask is not affected
        """
        super(AxialPlaneAffine, self).__init__(always_apply, p)

        self.scale_limit = M.prepare_non_negative_float(scale_limit, 'scale_limit')
        self.shift_limit = M.prepare_non_negative_float(shift_limit, 'shift_limit')
        self.angle_limit = M.prepare_inrange_zero_one_float(angle_limit / 180, 'angle_limit')

        self.border_mode = border_mode
        self.interpolation = interpolation
        self.mask_interpolation = E.InterType.NEAREST

        self.fill_value = M.prepare_float(fill_value, 'fill_value')
        self.mask_fill_value = M.prepare_float(mask_fill_value, 'mask_fill_value')

    def get_transform_init_args_names(self):
        return (
            'scale_limit',
            'shift_limit',
            'angle_limit',
            'border_mode',
            'interpolation',
            'fill_value',
            'mask_fill_value'
        )

    def get_params(self):
        scale = 1 + (2 * np.random.random() - 1) * self.scale_limit
        shift = (2 * np.random.random(C.NUM_PLANAR_DIMENSIONS) - 1) * self.shift_limit
        angle = 180 * (2 * np.random.random() - 1) * self.angle_limit

        return {'scale': scale, 'shift': shift, 'angle': angle}

    @property
    def targets_as_params(self):
        return ['voxel']

    def get_params_dependent_on_targets(self, params):
        shape = params['voxel'].shape[:C.NUM_SPATIAL_DIMENSIONS]

        return {'shape': shape}

    def apply(self, voxel, scale, shift, angle, **params):
        return FV.plane_affine(voxel, scale, shift, angle, self.interpolation, self.border_mode, self.fill_value, C.AXIAL_DIM)

    def apply_to_mask(self, mask, scale, shift, angle, **params):
        return FV.plane_affine(mask, scale, shift, angle, self.mask_interpolation, self.border_mode, self.mask_fill_value, C.AXIAL_DIM)

    def apply_to_points(self, points, scale, shift, angle, shape, **params):
        return FG.plane_affine(points, scale, shift, angle, C.AXIAL_DIM, shape)

class AxialPlaneScale(AxialPlaneAffine):
    """Randomly scale axial planes of a voxel.
    """
    def __init__(
            self,
            scale_limit=0.05,
            border_mode=E.BorderType.DEFAULT,
            interpolation=E.InterType.DEFAULT,
            fill_value=0,
            mask_fill_value=0,
            always_apply=False,
            p=0.5,
        ):
        """
            :args:
                scale_limit: float
                    limit of scaling
                border_mode: BorderType
                    border mode
                interpolation: InterType
                    interpolation mode
                fill_value: float
                    padding value of voxel if border_mode is cv2.BORDER_CONSTANT
                mask_fill_value: int or None
                    padding value if border_mode is cv2.BORDER_CONSTANT. if value is None, mask is not affected
        """
        super(AxialPlaneScale, self).__init__(scale_limit, 0, 0, border_mode, interpolation, fill_value, mask_fill_value, always_apply, p)

    def get_transform_init_args_names(self):
        return (
            'scale_limit',
            'border_mode',
            'interpolation',
            'fill_value',
            'mask_fill_value'
        )

class AxialPlaneTranslate(AxialPlaneAffine):
    """Randomly translate axial planes of a voxel.
    """
    def __init__(
            self,
            shift_limit=0.05,
            border_mode=E.BorderType.DEFAULT,
            interpolation=E.InterType.DEFAULT,
            fill_value=0,
            mask_fill_value=0,
            always_apply=False,
            p=0.5,
        ):
        """
            :args:
                shift_limit: float
                    limit of translation as ratio of size
                border_mode: BorderType
                    border mode
                interpolation: InterType
                    interpolation mode
                fill_value: float
                    padding value of voxel if border_mode is cv2.BORDER_CONSTANT
                mask_fill_value: int or None
                    padding value if border_mode is cv2.BORDER_CONSTANT. if value is None, mask is not affected
        """
        super(AxialPlaneTranslate, self).__init__(0, shift_limit, 0, border_mode, interpolation, fill_value, mask_fill_value, always_apply, p)

    def get_transform_init_args_names(self):
        return (
            'shift_limit',
            'border_mode',
            'interpolation',
            'fill_value',
            'mask_fill_value'
        )

class AxialPlaneRotate(AxialPlaneAffine):
    """Randomly rotate axial planes of a voxel.
    """
    def __init__(
            self,
            angle_limit=10,
            border_mode=E.BorderType.DEFAULT,
            interpolation=E.InterType.DEFAULT,
            fill_value=0,
            mask_fill_value=0,
            always_apply=False,
            p=0.5,
        ):
        """
            :args:
                angle_limit: float
                    limit of rotation in degrees [0, 180]
                border_mode: BorderType
                    border mode
                interpolation: InterType
                    interpolation mode
                fill_value: float
                    padding value of voxel if border_mode is cv2.BORDER_CONSTANT
                mask_fill_value: int or None
                    padding value if border_mode is cv2.BORDER_CONSTANT. if value is None, mask is not affected
        """
        super(AxialPlaneRotate, self).__init__(0, 0, angle_limit, border_mode, interpolation, fill_value, mask_fill_value, always_apply, p)

    def get_transform_init_args_names(self):
        return (
            'angle_limit',
            'border_mode',
            'interpolation',
            'fill_value',
            'mask_fill_value'
        )

class GaussNoise(VoxelOnlyTransform):
    """Randomly add gaussian noise to a voxel.
    """
    def __init__(
            self,
            mean=0.,
            variance=15.,
            per_channel=True,
            always_apply=False,
            p=0.5,
        ):
        """
            :args:
                mean: float
                    mean of gaussian noise
                variance: float
                    variance of gaussian noise
                per_channel: bool
                    if set to True, noise will be sampled for each channel independently
        """
        super(GaussNoise, self).__init__(always_apply, p)

        self.mean = M.prepare_float(mean, 'mean')
        self.variance = M.prepare_non_negative_float(variance, 'variance')
        self.per_channel = per_channel

    def get_transform_init_args_names(self):
        return ('mean', 'variance', 'per_channel')

    @property
    def targets_as_params(self):
        return ['voxel']

    def get_params_dependent_on_targets(self, params):
        if self.per_channel and len(params['voxel'].shape) == C.NUM_MULTI_CHANNEL_DIMENSIONS:
            shape = params['voxel'].shape
        else:
            shape = params['voxel'].shape[:C.NUM_SPATIAL_DIMENSIONS]

        gauss = np.random.normal(self.mean, self.variance**0.5, shape)

        return {'gauss': gauss}

    def apply(self, voxel, gauss, **params):
        return FV.add(voxel, gauss)

class GaussBlur(VoxelOnlyTransform):
    """Blur by gaussian a voxel.
    """
    def __init__(
            self,
            variance=1.,
            kernel_size_range=(3, 5),
            always_apply=False,
            p=0.5,
        ):
        """
            :args:
                variance: float
                    variance of gaussian kernel
                kernel_size_range: (int, int)
                    range for select kernel size of blur filter
        """
        super(GaussBlur, self).__init__(always_apply, p)

        self.variance = M.prepare_non_negative_float(variance, 'variance')
        self.kernel_size_range = M.prepare_int_asymrange(kernel_size_range, 'kernel_size_range', 0)

        self.min_kernel_size = kernel_size_range[0]
        self.max_kernel_size = kernel_size_range[1]

        if self.min_kernel_size % 2 == 0 or self.max_kernel_size % 2 == 0:
            raise ValueError('Invalid range borders. Must be odd, but got: {}.'.format(kernel_size_range))

    def get_transform_init_args_names(self):
        return ('variance', 'kernel_size_range')

    def get_params(self):
        kernel_size = 2 * np.random.randint(self.min_kernel_size // 2, self.max_kernel_size // 2 + 1) + 1

        x = np.kron(np.arange(kernel_size), np.ones((kernel_size, kernel_size, 1)))
        y = np.moveaxis(x, 0, -1)
        z = np.moveaxis(y, 0, -1)

        distances = (x - kernel_size // 2 )**2 + (y - kernel_size // 2 )**2 + (z - kernel_size // 2 )**2
        kernel = np.exp( -0.5 * distances / self.variance )

        kernel = kernel / kernel.sum()

        return {'kernel': kernel}

    def apply(self, voxel, kernel, **params):
        return FV.conv(voxel, kernel, E.BorderType.CONSTANT, 0)

class IntensityShift(VoxelOnlyTransform):
    """Shift intensities of a voxel.
    """
    def __init__(
            self,
            shift_limit=10.,
            always_apply=False,
            p=0.5,
        ):
        """
            :NOTE:
                The augmentation is often referred to as additive brightness in other libraries.

            :args:
                shift_limit: float
                    limit of intensity shift
        """
        super(IntensityShift, self).__init__(always_apply, p)

        self.shift_limit = M.prepare_non_negative_float(shift_limit, 'shift_limit')

    def get_transform_init_args_names(self):
        return ('shift_limit', )

    def get_params(self):
        shift = (2 * np.random.random() - 1) * self.shift_limit

        shift = np.expand_dims(shift, [C.VERTICAL_DIM, C.HORIZONTAL_DIM, C.AXIAL_DIM])

        return {'shift': shift}

    def apply(self, voxel, shift, **params):
        return FV.add(voxel, shift)

class IntensityScale(VoxelOnlyTransform):
    """Scale intensities of a voxel.
    """
    def __init__(
            self,
            scale_limit=10.,
            always_apply=False,
            p=0.5,
        ):
        """
            :NOTE:
                The augmentation is often referred to as multiplicative brightness in other libraries.

            :args:
                scale_limit: float
                    limit of intensity scale
        """
        super(IntensityScale, self).__init__(always_apply, p)

        self.scale_limit = M.prepare_non_negative_float(scale_limit, 'scale_limit')

    def get_transform_init_args_names(self):
        return ('scale_limit', )

    @property
    def targets_as_params(self):
        return ['voxel']

    def get_params_dependent_on_targets(self, params):
        scale = 1 + (2 * np.random.random() - 1) * self.scale_limit

        return {'scale': scale}

    def apply(self, voxel, scale, **params):
        return FV.multiply(voxel, scale)

class Contrast(VoxelOnlyTransform):
    """Change contrast of voxel.
    """
    def __init__(
            self,
            contrast_limit=0.1,
            always_apply=False,
            p=0.5,
        ):
        """
            :NOTE:
                The definition of contrasting was taken from https://arxiv.org/pdf/1902.05396

            :args:
                contrast_limit: float
                    limit of contrasting
        """
        super(Contrast, self).__init__(always_apply, p)

        self.contrast_limit = M.prepare_non_negative_float(contrast_limit, 'contrast_limit')

    def get_transform_init_args_names(self):
        return ('contrast_limit', )

    def get_params(self):
        contrast = 1 + (2 * np.random.random() - 1) * self.contrast_limit

        return {'contrast': contrast}

    def apply(self, voxel, contrast, **params):
        return FV.contrast(voxel, contrast)

class Gamma(VoxelOnlyTransform):
    """Apply gamma transform to intensities of a voxel.
    """
    def __init__(
            self,
            gamma_range=(0.8, 1.2),
            always_apply=False,
            p=0.5,
        ):
        """
            :args:
                gamma_range: (float, float)
                    limit of intensity shift
        """
        super(Gamma, self).__init__(always_apply, p)

        self.gamma_range = M.prepare_float_asymrange(gamma_range, 'gamma_range', 0.)

        self.min_gamma = gamma_range[0]
        self.max_gamma = gamma_range[1]

    def get_transform_init_args_names(self):
        return ('gamma_range', )

    def get_params(self):
        gamma = np.random.uniform(self.min_gamma, self.max_gamma)

        return {'gamma': gamma}

    def apply(self, voxel, gamma, **params):
        return FV.gamma(voxel, gamma)

class GridDistort(DualTransform):
    """Randomly distort a voxel by grid.
    """
    def __init__(
            self,
            distort_limit=0.05,
            ncells=4,
            interpolation=E.InterType.DEFAULT,
            always_apply=False,
            p=0.5,
        ):
        """
            :args:
                distort_limit: float
                    limit of distortion
                ncells: int
                    grid size
                interpolation: InterType
                    interpolation mode
        """
        super(GridDistort, self).__init__(always_apply, p)

        self.distort_limit = M.prepare_non_negative_float(distort_limit, 'distort_limit')
        self.ncells = M.prepare_int(ncells, 'ncells')

        self.interpolation = interpolation
        self.mask_interpolation = E.InterType.NEAREST

    def get_transform_init_args_names(self):
        return ('distort_limit', 'ncells', 'interpolation')

    @property
    def targets_as_params(self):
        return ['voxel']

    def get_params_dependent_on_targets(self, params):
        shape = np.array(params['voxel'].shape[:C.NUM_SPATIAL_DIMENSIONS])

        func = lambda _: np.linspace(0., 1., self.ncells+1)
        cells = tuple(map(func, range(C.NUM_SPATIAL_DIMENSIONS)))

        if self.ncells > 1:
            for cell in cells:
                directions = np.random.choice([-1, 1], size=self.ncells-1)
                magnitudes = np.random.random(size=self.ncells-1) * self.distort_limit * 0.5

                cell[1:-1] += directions * magnitudes / (self.ncells+1)

        func = lambda size: np.linspace(0., 1., size)
        points = map(func, shape)

        func = lambda point, cell, size: size * np.interp(point, cell, np.linspace(0., 1., self.ncells+1))
        distorted_points = map(func, points, cells, shape-1)
        distorted_grid = np.meshgrid(*distorted_points, indexing='ij')

        return {'distorted_grid': distorted_grid}

    def apply(self, voxel, distorted_grid, **params):
        return FV.distort(voxel, distorted_grid, self.interpolation)

    def apply_to_mask(self, mask, distorted_grid, **params):
        return FV.distort(mask, distorted_grid, self.mask_interpolation)

class ElasticDistort(DualTransform):
    """Randomly elastic distort a voxel.
    """
    def __init__(
            self,
            distort_limit=0.05,
            sigma=1,
            interpolation=E.InterType.DEFAULT,
            always_apply=False,
            p=0.5,
        ):
        """
            :args:
                distort_limit: float
                    limit of distortion
                sigma: float
                    smoothness of transformation
                interpolation: InterType
                    interpolation mode
        """
        super(ElasticDistort, self).__init__(always_apply, p)

        self.distort_limit = M.prepare_non_negative_float(distort_limit, 'distort_limit')
        self.sigma = M.prepare_float(sigma, 'sigma')

        self.interpolation = interpolation
        self.mask_interpolation = E.InterType.NEAREST

    def get_transform_init_args_names(self):
        return ('distort_limit', 'sigma', 'interpolation')

    @property
    def targets_as_params(self):
        return ['voxel']

    def get_params_dependent_on_targets(self, params):
        shape = np.array(params['voxel'].shape[:C.NUM_SPATIAL_DIMENSIONS])

        func = lambda size: np.linspace(0., size-1, size)
        points = map(func, shape)

        grid = np.meshgrid(*points, indexing='ij')
        distorted_grid = grid

        return {'distorted_grid': distorted_grid}

    def apply(self, voxel, distorted_grid, **params):
        return FV.distort(voxel, distorted_grid, self.interpolation)

    def apply_to_mask(self, mask, distorted_grid, **params):
        return FV.distort(mask, distorted_grid, self.mask_interpolation)

class PlaneDropout(VoxelOnlyTransform):
    """Randomly drop out planes of a voxel along a dim.
    """
    _DIMS = (C.HORIZONTAL_DIM, C.VERTICAL_DIM, C.AXIAL_DIM)

    def __init__(
            self,
            nplanes=3,
            fill_value=0.,
            always_apply=False,
            p=0.5,
        ):
        """
            :args:
                nplanes: int
                    number of dropping planes
                fill_value: float
                    filling value of voxel
        """
        super(PlaneDropout, self).__init__(always_apply, p)

        self.nplanes = M.prepare_non_negative_int(nplanes, 'nplanes')

        self.fill_value = M.prepare_float(fill_value, 'fill_value')

    def get_transform_init_args_names(self):
        return ('nplanes', 'fill_value')

    @property
    def targets_as_params(self):
        return ['voxel']

    def get_params_dependent_on_targets(self, params):
        dim = np.random.choice(self._DIMS)
        size = params['voxel'].shape[dim]

        indices = np.random.choice(size, size=self.nplanes, replace=False)

        return {'indices': indices, 'dim': dim}

    def apply(self, voxel, indices, dim, **params):
        return FV.plane_dropout(voxel, indices, self.fill_value, dim)

class HorizontalPlaneDropout(PlaneDropout):
    """Randomly drop out horizontal planes of a voxel.
    """
    _DIMS = (C.HORIZONTAL_DIM, )

class VerticalPlaneDropout(PlaneDropout):
    """Randomly drop out vertical planes of a voxel.
    """
    _DIMS = (C.VERTICAL_DIM, )

class AxialPlaneDropout(PlaneDropout):
    """Randomly drop out axial planes of a voxel.
    """
    _DIMS = (C.AXIAL_DIM, )

class PatchDropout(VoxelOnlyTransform):
    """Randomly drop out patches of a voxel.
    """
    def __init__(
            self,
            npatches=3,
            height_range=(8, 8),
            width_range=(8, 8),
            depth_range=(8, 8),
            fill_value=0.,
            always_apply=False,
            p=0.5,
        ):
        """
            :args:
                npatches: int
                    number of dropping pathes
                height_range: (int, int)
                     range for selecting of the height of patch
                width_range: (int, int)
                     range for selecting of the width of patch
                depth_range: (int, int)
                     range for selecting of the depth of patch
                fill_value: float
                    filling value of voxel
        """
        super(PatchDropout, self).__init__(always_apply, p)

        self.npatches = M.prepare_non_negative_int(npatches, 'npatches')

        self.height_range = M.prepare_int_asymrange(height_range, 'height_range', 0)

        self.min_height_range = height_range[0]
        self.max_height_range = height_range[1]

        self.width_range = M.prepare_int_asymrange(width_range, 'width_range', 0)

        self.min_width_range = width_range[0]
        self.max_width_range = width_range[1]

        self.depth_range = M.prepare_int_asymrange(depth_range, 'depth_range', 0)

        self.min_depth_range = depth_range[0]
        self.max_depth_range = depth_range[1]

        self.fill_value = M.prepare_float(fill_value, 'fill_value')

    def get_transform_init_args_names(self):
        return (
            'npatches',
            'height_range',
            'width_range',
            'depth_range',
            'fill_value'
        )

    @property
    def targets_as_params(self):
        return ['voxel']

    def get_params_dependent_on_targets(self, params):
        shape = params['voxel'].shape[:C.NUM_SPATIAL_DIMENSIONS]

        patch_shape_ranges = (
            (self.min_height_range, self.max_height_range + 1),
            (self.min_width_range, self.max_width_range + 1),
            (self.min_depth_range, self.max_depth_range + 1),
        )

        patches = []

        for _ in range(self.npatches):
            selector = tuple()

            for size, patch_size_range in zip(shape, patch_shape_ranges):
                patch_size = np.random.randint(*patch_size_range)
                start = np.random.randint(0, size - patch_size + 1)
                end = start + patch_size

                selector = (*selector, slice(start, end))

            patches.append(selector)

        return {'patches': patches}

    def apply(self, voxel, patches, **params):
        return FV.patch_dropout(voxel, patches, self.fill_value)

class PatchShuffle(VoxelOnlyTransform):
    """Randomly shuffle pixels in patches of a voxel.
    """
    def __init__(
            self,
            npatches=3,
            height_range=(8, 8),
            width_range=(8, 8),
            depth_range=(8, 8),
            always_apply=False,
            p=0.5,
        ):
        """
            :args:
                npatches: int
                    number of dropping pathes
                height_range: (int, int)
                     range for selecting of the height of patch
                width_range: (int, int)
                     range for selecting of the width of patch
                depth_range: (int, int)
                     range for selecting of the depth of patch
        """
        super(PatchShuffle, self).__init__(always_apply, p)

        self.npatches = M.prepare_non_negative_int(npatches, 'npatches')

        self.height_range = M.prepare_int_asymrange(height_range, 'height_range', 0)

        self.min_height_range = height_range[0]
        self.max_height_range = height_range[1]

        self.width_range = M.prepare_int_asymrange(width_range, 'width_range', 0)

        self.min_width_range = width_range[0]
        self.max_width_range = width_range[1]

        self.depth_range = M.prepare_int_asymrange(depth_range, 'depth_range', 0)

        self.min_depth_range = depth_range[0]
        self.max_depth_range = depth_range[1]

    def get_transform_init_args_names(self):
        return (
            'npatches',
            'height_range',
            'width_range',
            'depth_range'
        )

    @property
    def targets_as_params(self):
        return ['voxel']

    def get_params_dependent_on_targets(self, params):
        shape = params['voxel'].shape[:C.NUM_SPATIAL_DIMENSIONS]

        patch_shape_ranges = (
            (self.min_height_range, self.max_height_range + 1),
            (self.min_width_range, self.max_width_range + 1),
            (self.min_depth_range, self.max_depth_range + 1),
        )

        patches = []

        for _ in range(self.npatches):
            selector = tuple()

            for size, patch_size_range in zip(shape, patch_shape_ranges):
                patch_size = np.random.randint(*patch_size_range)
                start = np.random.randint(0, size - patch_size + 1)
                end = start + patch_size

                selector = (*selector, slice(start, end))

            patches.append(selector)

        return {'patches': patches}

    def apply(self, voxel, patches, **params):
        return FV.patch_shuffle(voxel, patches)

class Downscale(VoxelOnlyTransform):
    """Randomly reduce the resolution of voxel.
    """
    def __init__(
            self,
            scale_limit=0.25,
            interpolation=E.InterType.NEAREST,
            always_apply=False,
            p=0.5,
        ):
        """
            :args:
                scale_limit: float
                    limit of resolution scale
                interpolation: InterType
                    interpolation mode for downscale
        """
        super(Downscale, self).__init__(always_apply, p)

        self.scale_limit = M.prepare_non_negative_float(scale_limit, 'scale_limit')

        self.down_interpolation = interpolation
        self.up_interpolation = E.InterType.LINEAR

    def get_transform_init_args_names(self):
        return (
            'scale_limit',
            'down_interpolation',
        )

    def get_params(self):
        scale = 1 + (np.random.random() - 1) * self.scale_limit

        return {'scale': scale}

    def apply(self, voxel, scale, **params):
        return FV.downscale(voxel, scale, self.down_interpolation, self.up_interpolation)
