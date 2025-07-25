import numpy as np

import voxelmentations.core.enum as E
import voxelmentations.core.constants as C
import voxelmentations.augmentations.checkers as M

from voxelmentations.augmentations.functional import FV, FG
from voxelmentations.core import VoxelOnlyAugmentation, DualAugmentation, register_as_serializable

@register_as_serializable
class PadIfNeeded(DualAugmentation):
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

    def get_augmentation_init_args_names(self):
        return (
            'min_height',
            'min_width',
            'min_depth',
            'position',
            'border_mode',
            'fill_value',
            'mask_fill_value',
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

    def apply_to_points(self, points, pads, **data):
        raise FG.pad(points, pads)

@register_as_serializable
class Affine(DualAugmentation):
    """Randomly deform a voxel.
    """
    def __init__(
            self,
            scale_limit=0.05,
            angle_limit=10,
            shift_limit=0.05,
            border_mode=E.BorderType.DEFAULT,
            interpolation=E.InterType.DEFAULT,
            fill_value=0,
            mask_fill_value=0,
            isotropic=True,
            always_apply=False,
            p=0.5,
        ):
        """
            :args:
                scale_limit: float
                    limit of scaling
                angle_limit: float
                    limit of rotation in degrees [0, 180]
                shift_limit: float
                    limit of translation as ratio of size
                border_mode: BorderType
                    border mode
                interpolation: InterType
                    interpolation mode
                fill_value: float
                    padding value of voxel if border_mode is BorderType.CONSTANT
                mask_fill_value: int or None
                    padding value if border_mode is BorderType.CONSTANT. if value is None, mask is not affected
                isotropic: bool
                    flag indicating whether the scale is isotropic or not
        """
        super(Affine, self).__init__(always_apply, p)

        self.scale_limit = M.prepare_non_negative_float(scale_limit, 'scale_limit')
        self.angle_limit = 180 * M.prepare_inrange_zero_one_float(angle_limit / 180, 'angle_limit')
        self.shift_limit = M.prepare_non_negative_float(shift_limit, 'shift_limit')

        self.border_mode = border_mode
        self.interpolation = interpolation
        self.mask_interpolation = E.InterType.NEAREST

        self.fill_value = M.prepare_float(fill_value, 'fill_value')
        self.mask_fill_value = M.prepare_float(mask_fill_value, 'mask_fill_value')

        self.isotropic = isotropic

    def get_augmentation_init_args_names(self):
        return (
            'scale_limit',
            'angle_limit',
            'shift_limit',
            'border_mode',
            'interpolation',
            'fill_value',
            'mask_fill_value',
            'isotropic',
        )

    def get_params(self):
        scale = 1 + (2 * np.random.random(1 if self.isotropic else C.NUM_SPATIAL_DIMENSIONS) - 1) * self.scale_limit
        scale = scale.repeat(C.NUM_SPATIAL_DIMENSIONS) if self.isotropic else scale
        angles = (2 * np.random.random(C.NUM_SPATIAL_DIMENSIONS) - 1) * self.angle_limit
        shift = (2 * np.random.random(C.NUM_SPATIAL_DIMENSIONS) - 1) * self.shift_limit

        return {'scale': scale, 'angles': angles, 'shift': shift}

    @property
    def targets_as_params(self):
        return ['voxel']

    def get_params_dependent_on_targets(self, params):
        shape = params['voxel'].shape[:C.NUM_SPATIAL_DIMENSIONS]

        return {'shape': shape}

    def apply(self, voxel, scale, angles, shift, **params):
        return FV.affine(voxel, scale, angles, shift, self.interpolation, self.border_mode, self.fill_value)

    def apply_to_mask(self, mask, scale, angles, shift, **params):
        return FV.affine(mask, scale, angles, shift, self.mask_interpolation, self.border_mode, self.mask_fill_value)

    def apply_to_points(self, points, scale, angles, shift, shape, **params):
        return FG.affine(points, scale, angles, shift, shape)

@register_as_serializable
class Scale(Affine):
    """Randomly scale a voxel.
    """
    def __init__(
            self,
            scale_limit=0.05,
            border_mode=E.BorderType.DEFAULT,
            interpolation=E.InterType.DEFAULT,
            fill_value=0,
            mask_fill_value=0,
            isotropic=True,
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
                    padding value of voxel if border_mode is BorderType.CONSTANT
                mask_fill_value: int or None
                    padding value if border_mode is BorderType.CONSTANT. if value is None, mask is not affected
                isotropic: bool
                    flag indicating whether the scale is isotropic or not
        """
        super(Scale, self).__init__(scale_limit, 0, 0, border_mode, interpolation, fill_value, mask_fill_value, isotropic, always_apply, p)

    def get_augmentation_init_args_names(self):
        return (
            'scale_limit',
            'border_mode',
            'interpolation',
            'fill_value',
            'mask_fill_value',
            'isotropic',
        )

@register_as_serializable
class Translate(Affine):
    """Randomly translate a voxel.
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
        super(Translate, self).__init__(0, 0, shift_limit, border_mode, interpolation, fill_value, mask_fill_value, True, always_apply, p)

    def get_augmentation_init_args_names(self):
        return (
            'shift_limit',
            'border_mode',
            'interpolation',
            'fill_value',
            'mask_fill_value',
        )

@register_as_serializable
class AxialPlaneAffine(DualAugmentation):
    """Randomly deform axial planes of a voxel.
    """
    def __init__(
            self,
            scale_limit=0.05,
            angle_limit=10,
            shear_limit=10,
            shift_limit=0.05,
            border_mode=E.BorderType.DEFAULT,
            interpolation=E.InterType.DEFAULT,
            fill_value=0,
            mask_fill_value=0,
            isotropic=True,
            always_apply=False,
            p=0.5,
        ):
        """
            :args:
                scale_limit: float
                    limit of scaling
                angle_limit: float
                    limit of rotation in degrees [0, 180]
                shear_limit: float
                    limit of shearing in degrees [0, 180]
                shift_limit: float
                    limit of translation as ratio of size
                border_mode: BorderType
                    border mode
                interpolation: InterType
                    interpolation mode
                fill_value: float
                    padding value of voxel if border_mode is BorderType.CONSTANT
                mask_fill_value: int or None
                    padding value if border_mode is BorderType.CONSTANT. if value is None, mask is not affected
                isotropic: bool
                    flag indicating whether the scale is isotropic or not
        """
        super(AxialPlaneAffine, self).__init__(always_apply, p)

        self.scale_limit = M.prepare_non_negative_float(scale_limit, 'scale_limit')
        self.angle_limit = 180 * M.prepare_inrange_zero_one_float(angle_limit / 180, 'angle_limit')
        self.shear_limit = 180 * M.prepare_inrange_zero_one_float(shear_limit / 180, 'shear_limit')
        self.shift_limit = M.prepare_non_negative_float(shift_limit, 'shift_limit')

        self.border_mode = border_mode
        self.interpolation = interpolation
        self.mask_interpolation = E.InterType.NEAREST

        self.fill_value = M.prepare_float(fill_value, 'fill_value')
        self.mask_fill_value = M.prepare_float(mask_fill_value, 'mask_fill_value')

        self.isotropic = isotropic

    def get_augmentation_init_args_names(self):
        return (
            'scale_limit',
            'angle_limit',
            'shear_limit',
            'shift_limit',
            'border_mode',
            'interpolation',
            'fill_value',
            'mask_fill_value',
            'isotropic',
        )

    def get_params(self):
        scale = 1 + (2 * np.random.random(1 if self.isotropic else C.NUM_PLANAR_DIMENSIONS) - 1) * self.scale_limit
        scale = scale.repeat(C.NUM_PLANAR_DIMENSIONS) if self.isotropic else scale
        angle = (2 * np.random.random() - 1) * self.angle_limit
        shear = (2 * np.random.random(C.NUM_PLANAR_DIMENSIONS) - 1) * self.shear_limit
        shift = (2 * np.random.random(C.NUM_PLANAR_DIMENSIONS) - 1) * self.shift_limit

        return {'scale': scale, 'angle': angle, 'shear': shear, 'shift': shift}

    @property
    def targets_as_params(self):
        return ['voxel']

    def get_params_dependent_on_targets(self, params):
        shape = params['voxel'].shape[:C.NUM_SPATIAL_DIMENSIONS]

        return {'shape': shape}

    def apply(self, voxel, scale, angle, shear, shift, **params):
        return FV.plane_affine(voxel, scale, angle, shear, shift, self.interpolation, self.border_mode, self.fill_value, C.AXIAL_DIM)

    def apply_to_mask(self, mask, scale, angle, shear, shift, **params):
        return FV.plane_affine(mask, scale, angle, shear, shift, self.mask_interpolation, self.border_mode, self.mask_fill_value, C.AXIAL_DIM)

    def apply_to_points(self, points, scale, angle, shear, shift, shape, **params):
        return FG.plane_affine(points, scale, angle, shear, shift, C.AXIAL_DIM, shape)

@register_as_serializable
class AxialPlaneScale(AxialPlaneAffine):
    """Randomly scale axial planes of a voxel.
    """
    def __init__(
            self,
            scale_limit=0.1,
            border_mode=E.BorderType.DEFAULT,
            interpolation=E.InterType.DEFAULT,
            fill_value=0,
            mask_fill_value=0,
            isotropic=True,
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
                isotropic: bool
                    flag indicating whether the scale is isotropic or not
        """
        super(AxialPlaneScale, self).__init__(scale_limit, 0, 0, 0, border_mode, interpolation, fill_value, mask_fill_value, isotropic, always_apply, p)

    def get_augmentation_init_args_names(self):
        return (
            'scale_limit',
            'border_mode',
            'interpolation',
            'fill_value',
            'mask_fill_value',
            'isotropic',
        )

@register_as_serializable
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
        super(AxialPlaneRotate, self).__init__(0, angle_limit, 0, 0, border_mode, interpolation, fill_value, mask_fill_value, True, always_apply, p)

    def get_augmentation_init_args_names(self):
        return (
            'angle_limit',
            'border_mode',
            'interpolation',
            'fill_value',
            'mask_fill_value',
        )

@register_as_serializable
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
        super(AxialPlaneTranslate, self).__init__(0, 0, 0, shift_limit, border_mode, interpolation, fill_value, mask_fill_value, True, always_apply, p)

    def get_augmentation_init_args_names(self):
        return (
            'shift_limit',
            'border_mode',
            'interpolation',
            'fill_value',
            'mask_fill_value',
        )

@register_as_serializable
class GaussNoise(VoxelOnlyAugmentation):
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

    def get_augmentation_init_args_names(self):
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

@register_as_serializable
class GaussBlur(VoxelOnlyAugmentation):
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

    def get_augmentation_init_args_names(self):
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

@register_as_serializable
class IntensityShift(VoxelOnlyAugmentation):
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

    def get_augmentation_init_args_names(self):
        return ('shift_limit', )

    def get_params(self):
        shift = (2 * np.random.random() - 1) * self.shift_limit

        shift = np.expand_dims(shift, [C.VERTICAL_DIM, C.HORIZONTAL_DIM, C.AXIAL_DIM])

        return {'shift': shift}

    def apply(self, voxel, shift, **params):
        return FV.add(voxel, shift)

@register_as_serializable
class IntensityScale(VoxelOnlyAugmentation):
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

    def get_augmentation_init_args_names(self):
        return ('scale_limit', )

    @property
    def targets_as_params(self):
        return ['voxel']

    def get_params_dependent_on_targets(self, params):
        scale = 1 + (2 * np.random.random() - 1) * self.scale_limit

        return {'scale': scale}

    def apply(self, voxel, scale, **params):
        return FV.multiply(voxel, scale)

@register_as_serializable
class Contrast(VoxelOnlyAugmentation):
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

    def get_augmentation_init_args_names(self):
        return ('contrast_limit', )

    def get_params(self):
        contrast = 1 + (2 * np.random.random() - 1) * self.contrast_limit

        return {'contrast': contrast}

    def apply(self, voxel, contrast, **params):
        return FV.contrast(voxel, contrast)

@register_as_serializable
class Gamma(VoxelOnlyAugmentation):
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

    def get_augmentation_init_args_names(self):
        return ('gamma_range', )

    def get_params(self):
        gamma = np.random.uniform(self.min_gamma, self.max_gamma)

        return {'gamma': gamma}

    def apply(self, voxel, gamma, **params):
        return FV.gamma(voxel, gamma)

@register_as_serializable
class GridDistort(DualAugmentation):
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

    def get_augmentation_init_args_names(self):
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

    def apply_to_points(self, points, distorted_grid, **data):
        raise NotImplemented()

@register_as_serializable
class ElasticDistort(DualAugmentation):
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

    def get_augmentation_init_args_names(self):
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

    def apply_to_points(self, points, distorted_grid, **data):
        raise NotImplemented()

@register_as_serializable
class PlaneDropout(VoxelOnlyAugmentation):
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

    def get_augmentation_init_args_names(self):
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

@register_as_serializable
class HorizontalPlaneDropout(PlaneDropout):
    """Randomly drop out horizontal planes of a voxel.
    """
    _DIMS = (C.HORIZONTAL_DIM, )

@register_as_serializable
class VerticalPlaneDropout(PlaneDropout):
    """Randomly drop out vertical planes of a voxel.
    """
    _DIMS = (C.VERTICAL_DIM, )

@register_as_serializable
class AxialPlaneDropout(PlaneDropout):
    """Randomly drop out axial planes of a voxel.
    """
    _DIMS = (C.AXIAL_DIM, )

@register_as_serializable
class PatchDropout(VoxelOnlyAugmentation):
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

    def get_augmentation_init_args_names(self):
        return (
            'npatches',
            'height_range',
            'width_range',
            'depth_range',
            'fill_value',
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

@register_as_serializable
class PatchShuffle(VoxelOnlyAugmentation):
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

    def get_augmentation_init_args_names(self):
        return (
            'npatches',
            'height_range',
            'width_range',
            'depth_range',
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

@register_as_serializable
class Downscale(VoxelOnlyAugmentation):
    """Randomly reduce the resolution of voxel.
    """
    def __init__(
            self,
            scale_limit=0.25,
            downscale_interpolation=E.InterType.NEAREST,
            always_apply=False,
            p=0.5,
        ):
        """
            :NOTE:
                interpolation mode for upscale is preset to NEAREST

            :args:
                scale_limit: float
                    limit of resolution scale
                downscale_interpolation: InterType
                    interpolation mode for downscale
        """
        super(Downscale, self).__init__(always_apply, p)

        self.scale_limit = M.prepare_non_negative_float(scale_limit, 'scale_limit')

        self.downscale_interpolation = downscale_interpolation
        self.upscale_interpolation = E.InterType.LINEAR

    def get_augmentation_init_args_names(self):
        return (
            'scale_limit',
            'downscale_interpolation',
        )

    def get_params(self):
        scale = 1 + (np.random.random() - 1) * self.scale_limit

        return {'scale': scale}

    def apply(self, voxel, scale, **params):
        return FV.downscale(voxel, scale, self.downscale_interpolation, self.upscale_interpolation)
