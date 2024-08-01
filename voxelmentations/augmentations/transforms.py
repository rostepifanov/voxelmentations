import cv2
import numpy as np

import voxelmentations.core.enum as E
import voxelmentations.core.constants as C
import voxelmentations.augmentations.misc as M
import voxelmentations.augmentations.functional as F

from voxelmentations.core.transforms import VoxelOnlyTransform, DualTransform

class PadIfNeeded(DualTransform):
    """Pad shape of the voxel to the minimal shape.
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

    def apply(self, voxel, pads, **params):
        return F.pad(voxel, pads, self.border_mode, self.fill_value)

    def apply_to_mask(self, mask, pads, **params):
        return F.pad(mask, pads, self.border_mode, self.mask_fill_value)

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

    def get_transform_init_args_names(self):
        return ('min_height', 'min_width', 'min_depth', 'position', 'border_mode', 'fill_value', 'mask_fill_value')

class Flip(DualTransform):
    """Flip the input voxel along a dim.
    """
    _DIMS = (C.HORIZONTAL_DIM, C.VERTICAL_DIM, C.AXIAL_DIM)

    def apply(self, voxel, dims, **params):
        return F.flip(voxel, dims)

    def get_params(self):
        code = np.random.randint(1, 2**len(self._DIMS))

        dims = tuple()

        for dim in self._DIMS:
            if code % 2 == 1:
                dims = (*dims, dim)

            code = code / 2

        return {'dims': dims}

    def get_transform_init_args_names(self):
        return tuple()

class AxialFlip(Flip):
    """Flip the input voxel in z dim.
    """
    _DIMS = (C.AXIAL_DIM, )

class AxialPlaneFlip(Flip):
    """Flip the input voxel in x-y plane.
    """
    _DIMS = (C.HORIZONTAL_DIM, C.VERTICAL_DIM)

class PlaneDropout(DualTransform):
    """Randomly drop out planes of input voxel along a dim.
    """
    _DIMS = (C.HORIZONTAL_DIM, C.VERTICAL_DIM, C.AXIAL_DIM)

    def __init__(
            self,
            dropout_rate=0.05,
            fill_value=0.,
            mask_fill_value=None,
            always_apply=False,
            p=0.5,
        ):
        """
            :args:
                dropout_rate: float
                    percent of dropped planes
                fill_value: float
                    padding value of voxel if border_mode is cv2.BORDER_CONSTANT
                mask_fill_value: int or None
                    padding value if border_mode is cv2.BORDER_CONSTANT. if value is None, mask is not affected
        """
        super(PlaneDropout, self).__init__(always_apply, p)

        self.dropout_rate = M.prepare_inrange_zero_one_float(dropout_rate, 'dropout_rate')

        self.fill_value = M.prepare_float(fill_value, 'fill_value')
        self.mask_fill_value = mask_fill_value

    def apply(self, voxel, indices, dim, **params):
        return F.plane_dropout(voxel, indices, self.fill_value, dim)

    def apply_to_mask(self, mask, indices, dim, **params):
        if self.mask_fill_value is None:
            return mask
        else:
            return F.plane_dropout(mask, indices, self.mask_fill_value, dim)

    @property
    def targets_as_params(self):
        return ['voxel']

    def get_params_dependent_on_targets(self, params):
        dim = np.random.choice(self._DIMS)

        shape = params['voxel'].shape[dim]
        size = int(shape*self.dropout_rate)

        indices = np.random.choice(shape, size=size, replace=False)

        return {'indices': indices, 'dim': dim}

    def get_transform_init_args_names(self):
        return ('dropout_rate', 'fill_value', 'mask_fill_value')

class HorizontalPlaneDropout(PlaneDropout):
    """Randomly drop out horizontal planes of input voxel.
    """
    _DIMS = (C.HORIZONTAL_DIM, )

class VerticalPlaneDropout(PlaneDropout):
    """Randomly drop out vertical planes of input voxel.
    """
    _DIMS = (C.VERTICAL_DIM, )

class AxialPlaneDropout(PlaneDropout):
    """Randomly drop out axial planes of input voxel.
    """
    _DIMS = (C.AXIAL_DIM, )

class AxialPlaneRotate(DualTransform):
    """Randomly rotate axial planes of input voxel.
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
        super(AxialPlaneRotate, self).__init__(always_apply, p)

        self.angle_limit = M.prepare_inrange_zero_one_float(angle_limit / 180, 'angle_limit')

        self.border_mode = border_mode
        self.interpolation = interpolation
        self.mask_interpolation = E.InterType.NEAREST

        self.fill_value = M.prepare_float(fill_value, 'fill_value')
        self.mask_fill_value = M.prepare_float(mask_fill_value, 'mask_fill_value')

    def apply(self, voxel, angle, **params):
        return F.plane_rotate(voxel, angle, self.interpolation, self.border_mode, self.fill_value, C.AXIAL_DIM)

    def apply_to_mask(self, mask, angle, **params):
        return F.plane_rotate(mask, angle, self.mask_interpolation, self.border_mode, self.mask_fill_value, C.AXIAL_DIM)

    def get_params(self):
        angle = 180 * (2 * np.random.random() - 1) * self.angle_limit

        return {'angle': angle}

    def get_transform_init_args_names(self):
        return ('angle_limit', 'border_mode', 'interpolation', 'fill_value', 'mask_fill_value')

class AxialPlaneScale(DualTransform):
    """Randomly scale axial planes of input voxel.
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
        super(AxialPlaneScale, self).__init__(always_apply, p)

        self.scale_limit = M.prepare_non_negative_float(scale_limit, 'scale_limit')

        self.border_mode = border_mode
        self.interpolation = interpolation
        self.mask_interpolation = E.InterType.NEAREST

        self.fill_value = M.prepare_float(fill_value, 'fill_value')
        self.mask_fill_value = M.prepare_float(mask_fill_value, 'mask_fill_value')

    def apply(self, voxel, scale, **params):
        return F.plane_scale(voxel, scale, self.interpolation, self.border_mode, self.fill_value, C.AXIAL_DIM)

    def apply_to_mask(self, mask, scale, **params):
        return F.plane_scale(mask, scale, self.mask_interpolation, self.border_mode, self.mask_fill_value, C.AXIAL_DIM)

    def get_params(self):
        scale = 1 + (2 * np.random.random() - 1) * self.scale_limit

        return {'scale': scale}

    def get_transform_init_args_names(self):
        return ('scale_limit', 'border_mode', 'interpolation', 'fill_value', 'mask_fill_value')

class AxialPlaneAffine(DualTransform):
    """Randomly deform axial planes of input voxel.
    """
    def __init__(
            self,
            angle_limit=10,
            shift_limit=0.05,
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
                angle_limit: float
                    limit of rotation in degrees [0, 180]
                shift_limit: float
                    limit of shifting
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
        super(AxialPlaneAffine, self).__init__(always_apply, p)

        self.angle_limit = M.prepare_inrange_zero_one_float(angle_limit / 180, 'angle_limit')
        self.shift_limit = M.prepare_non_negative_float(shift_limit, 'shift_limit')
        self.scale_limit = M.prepare_non_negative_float(scale_limit, 'scale_limit')

        self.border_mode = border_mode
        self.interpolation = interpolation
        self.mask_interpolation = E.InterType.NEAREST

        self.fill_value = M.prepare_float(fill_value, 'fill_value')
        self.mask_fill_value = M.prepare_float(mask_fill_value, 'mask_fill_value')

    def apply(self, voxel, angle, shift, scale, **params):
        return F.plane_affine(voxel, angle, shift, scale, self.interpolation, self.border_mode, self.fill_value, C.AXIAL_DIM)

    def apply_to_mask(self, mask, angle, shift, scale, **params):
        return F.plane_affine(mask, angle, shift, scale, self.mask_interpolation, self.border_mode, self.mask_fill_value, C.AXIAL_DIM)

    def get_params(self):
        angle = 180 * (2 * np.random.random() - 1) * self.angle_limit
        shift = (2 * np.random.random() - 1) * self.shift_limit
        scale = 1 + (2 * np.random.random() - 1) * self.scale_limit

        return {'angle': angle, 'shift': shift, 'scale': scale}

    def get_transform_init_args_names(self):
        return ('angle_limit', 'scale_limit', 'border_mode', 'interpolation', 'fill_value', 'mask_fill_value')

class GaussNoise(VoxelOnlyTransform):
    """Randomly add gaussian noise to the voxel.
    """
    def __init__(
            self,
            mean=0.,
            variance=15,
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

    def apply(self, voxel, gauss, **params):
        return F.addition(voxel, gauss)

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

    def get_transform_init_args_names(self):
        return ('mean', 'variance', 'per_channel')

class GaussBlur(VoxelOnlyTransform):
    """Blur by gaussian the voxel.
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

    def apply(self, voxel, kernel, **params):
        return F.conv(voxel, kernel, E.BorderType.CONSTANT, 0)

    def get_params(self):
        kernel_size = 2 * np.random.randint(self.min_kernel_size // 2, self.max_kernel_size // 2 + 1) + 1

        x = np.kron(np.arange(kernel_size), np.ones((kernel_size, kernel_size, 1)))
        y = np.moveaxis(x, 0, -1)
        z = np.moveaxis(y, 0, -1)

        distances = (x - kernel_size // 2 )**2 + (y - kernel_size // 2 )**2 + (z - kernel_size // 2 )**2
        kernel = np.exp( -0.5 * distances / self.variance )

        kernel = kernel / kernel.sum()

        return {'kernel': kernel}

    def get_transform_init_args_names(self):
        return ('variance', 'kernel_size_range')

class IntensityShift(VoxelOnlyTransform):
    """Shift intensities of the voxel.
    """
    def __init__(
            self,
            shift_limit=10.,
            per_channel=True,
            always_apply=False,
            p=0.5,
        ):
        """
            :args:
                shift_limit: float
                    limit of intensity shift
                per_channel: bool
                    if set to True, noise will be sampled for each channel independently
        """
        super(IntensityShift, self).__init__(always_apply, p)

        self.shift_limit = M.prepare_non_negative_float(shift_limit, 'shift_limit')
        self.per_channel = per_channel

    def apply(self, voxel, shift, **params):
        return F.addition(voxel, shift)

    @property
    def targets_as_params(self):
        return ['voxel']

    def get_params_dependent_on_targets(self, params):
        if self.per_channel and len(params['voxel'].shape) == C.NUM_MULTI_CHANNEL_DIMENSIONS:
            nchannel = params['voxel'].shape[C.CHANNEL_DIM]
            shift = (2 * np.random.random(nchannel) - 1) * self.shift_limit
        else:
            shift = (2 * np.random.random() - 1) * self.shift_limit

        shift = np.expand_dims(shift, [C.VERTICAL_DIM, C.HORIZONTAL_DIM, C.AXIAL_DIM])

        return {'shift': shift}

    def get_transform_init_args_names(self):
        return ('shift_limit', 'per_channel')

class GridDistort(DualTransform):
    """Randomly distort the voxel by grid.
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

    def apply(self, voxel, cells, **params):
        return F.grid_distort(voxel, self.ncells, cells, self.interpolation)

    def apply_to_mask(self, mask, cells, **params):
        return F.grid_distort(mask, self.ncells, cells, self.mask_interpolation)

    def get_params(self):
        func = lambda _: np.linspace(0., 1., self.ncells+1)
        cells = tuple(map(func, range(C.NUM_SPATIAL_DIMENSIONS)))

        if self.ncells > 1:
            for cell in cells:
                directions = np.random.choice([-1, 1], size=self.ncells-1)
                magnitudes = np.random.random(size=self.ncells-1) * self.distort_limit * 0.5

                cell[1:-1] += directions * magnitudes / (self.ncells+1)

        return {'cells': cells}

    def get_transform_init_args_names(self):
        return ('distort_limit', 'ncells', 'interpolation')
