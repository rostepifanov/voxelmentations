from voxelmentations.core.application import Apply
from voxelmentations.core.utils import format_args

class Transform(Apply):
    """Root class for single augmentations
    """
    def __init__(self, always_apply=False, p=0.5):
        """
            :args:
                always_apply: bool
                    the flag of force application
                p: float
                    the probability of application
        """
        super(Transform, self).__init__(always_apply, p)

        self._targets = self.targets

    def get_transform_init_args_names(self):
        raise NotImplementedError(
            'Class {} is not serializable because the `get_transform_init_args_names` method is not '
            'implemented'.format(self.get_class_name())
        )

    def get_transform_init_args(self):
        return {k: getattr(self, k) for k in self.get_transform_init_args_names()}

    def __call__(self, *args, force_apply=False, **data):
        """
            :args:
                force_apply: bool
                    the flag of force application
                data: dict
                    the data to make a transformation

            :return:
                dict of transformed data
        """
        if args:
            raise KeyError('You have to pass data to augmentations as named arguments, for example: aug(voxel=voxel)')

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
                you must specify targets in subclass

                for example: ('voxel', ) or ('voxel', 'mask')
        """
        raise NotImplementedError

    def add_targets(self, additional_targets):
        """Add targets to transform them the same way as one of existing targets.

            :NOTE:
                for example: additional_targets = {'voxel2': 'voxel'}
                adds new target voxel2 that transformed the same way as voxel

            :args:
                additional_targets: dict
                    keys - additional target name, values - existed target name.
        """
        _targets = dict()

        for additional_key, existed_key in additional_targets.items():
            if additional_key in self._targets:
                existed_keys = ' '.join(self._targets.keys())

                raise ValueError(
                    f'Trying to overwrite existed target. '
                    f'Key={additional_key} exists in {existed_keys}',
                )
            elif existed_key in self._targets:
                _targets[additional_key] = self._targets[existed_key]

        self._targets.update(_targets)

    def _get_target_function(self, name):
        target_function = self._targets.get(name, lambda x, **p: x)
        return target_function

    def apply_with_params(self, params, **data):
        if params is None:
            return data

        pdata = {}

        for name, datum in data.items():
            if datum is not None:
                target_function = self._get_target_function(name)
                pdata[name] = target_function(datum, **params)
            else:
                pdata[name] = None

        return pdata

    def __repr__(self):
        state = self.get_base_init_args()
        state.update(self.get_transform_init_args())

        name = self.get_class_name()
        args=format_args(state)

        return '{}({})'.format(name, args)

class VoxelOnlyTransform(Transform):
    """Transform applied to voxel only
    """
    def apply(self, voxel, **params):
        raise NotImplementedError

    @property
    def targets(self):
        return {'voxel': self.apply}

class DualTransform(Transform):
    """Transform for segmentation task
    """
    def apply_to_mask(self, mask, **params):
        return self.apply(mask, **params)

    @property
    def targets(self):
        return {'voxel': self.apply, 'mask': self.apply_to_mask}

class Identity(DualTransform):
    """Identity transform
    """
    def get_transform_init_args_names(self):
        return tuple()

    def apply(self, voxel, **params):
        return voxel
