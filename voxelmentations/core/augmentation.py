from voxelmentations.core.utils import format_args
from voxelmentations.core.transformation import Transformation
from voxelmentations.core.serializable import register_as_serializable

class Augmentation(Transformation):
    """Root class for single augmentations
    """
    def __init__(self, always_apply=False, p=0.5):
        """
            :NOTE:
                the keys and the target are not the same
                the keys are extendable, but targets are predefined

                the data under "key" is transformed with corresponding "target" function

            :args:
                always_apply: bool
                    the flag of force application
                p: float
                    the probability of application
        """
        super(Augmentation, self).__init__(always_apply, p)

        self._keys = {k: v for k, v in  zip(self.targets.keys(), self.targets.keys()) }

    def get_augmentation_init_args_names(self):
        raise NotImplementedError(
            'Class {} is not serializable because the `get_augmentation_init_args_names` method is not '
            'implemented'.format(self.get_class_name())
        )

    def get_augmentation_init_args(self):
        return {k: getattr(self, k) for k in self.get_augmentation_init_args_names()}

    def get_state_dict(self):
        state = super().get_state_dict()

        state.update(self.get_class_fullname_as_dict())
        state.update(self.get_augmentation_init_args())

        return state

    def __call__(self, *args, force_apply=False, **data):
        """
            :args:
                force_apply: bool
                    the flag of force application
                data: dict
                    the data to augment

            :return:
                dict of augmentationed data
        """
        if args:
            raise KeyError('You have to pass data to augmentation as named arguments, for example: aug(voxel=voxel)')

        if self.whether_apply(force_apply):
            params = self.get_params()

            if self.targets_as_params:
                assert all(name in data for name in self.targets_as_params), '{} requires {}'.format(
                    self.get_class_name(), self.targets_as_params
                )

                targets_as_params = {name: data[name] for name in self.targets_as_params}

                params_dependent_on_targets = self.get_params_dependent_on_targets(targets_as_params)
                params.update(params_dependent_on_targets)

            return self.apply_with_params(params, **data)

        return data

    def get_params(self):
        return {}

    @property
    def targets_as_params(self):
        return []

    def get_params_dependent_on_targets(self, params):
        raise NotImplementedError(
            'Method get_params_dependent_on_targets is not implemented in class {}'.format(self.get_class_name())
        )

    @property
    def targets(self):
        """
            :NOTE:
                you must specify targets and their functions in subclass

                for example: ('voxel', ) or ('voxel', 'mask')
        """
        raise NotImplementedError

    def keys(self):
        """Return the keys that are transformed by target functions
        """
        return set(self._keys)

    def add_keys(self, additional_keys):
        """Add keys to augment them the same way as one of existing targets.

            :NOTE:
                for example: additional_keys = {'voxel2': 'voxel'}
                adds new key "voxel2" that transformed by "voxel" target function

            :args:
                additional_keys: dict
                    keys - additional key name, values - existed target name.
        """
        _keys = dict()

        for additional_key, existed_target in additional_keys.items():
            if additional_key in self._keys:
                existed_keys = ', '.join(self._keys)

                raise ValueError(
                    f'Trying to overwrite existed key. '
                    f'Key={additional_key} exists in [{existed_keys}]',
                )
            elif existed_target in self.targets.keys():
                _keys[additional_key] = self._keys[existed_target]

        self._keys.update(_keys)

        return self

    def _get_target_function(self, key):
        if key in self._keys:
            target = self._keys[key]
            function = self.targets[target]
        else:
            function = lambda x, **p: x

        return function

    def apply_with_params(self, params, **data):
        if params is None:
            return data

        pdata = {}

        for key, datum in data.items():
            if datum is not None:
                target_function = self._get_target_function(key)
                pdata[key] = target_function(datum, **params)
            else:
                pdata[key] = None

        return pdata

    def __repr__(self):
        state = self.get_base_init_args()
        state.update(self.get_augmentation_init_args())

        name = self.get_class_name()
        args=format_args(state)

        return '{}({})'.format(name, args)

class VoxelOnlyAugmentation(Augmentation):
    """Augmentation applied to voxel only
    """
    def apply(self, voxel, **params):
        raise NotImplementedError

    @property
    def targets(self):
        return {'voxel': self.apply}

class DualAugmentation(Augmentation):
    """Augmentation applied to voxel and segmentation mask
    """
    def apply_to_mask(self, mask, **params):
        return self.apply(mask, **params)

    @property
    def targets(self):
        return {
            'voxel': self.apply,
            'mask': self.apply_to_mask,
        }

class TripleAugmentation(Augmentation):
    """Augmentation applied to voxel, segmentation mask and key points
    """
    def apply_to_mask(self, mask, **params):
        return self.apply(mask, **params)

    @property
    def targets(self):
        return {
            'voxel': self.apply,
            'mask': self.apply_to_mask,
            'points': self.apply_to_points,
        }

@register_as_serializable
class Identity(TripleAugmentation):
    """Identity augmentation
    """
    def get_augmentation_init_args_names(self):
        return tuple()

    def apply(self, voxel, **params):
        return voxel

    def apply_to_points(self, points, **params):
        return points
