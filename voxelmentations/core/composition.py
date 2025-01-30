import numpy as np

from voxelmentations.core.utils import format_args
from voxelmentations.core.transformation import Transformation
from voxelmentations.core.serializable import register_as_serializable

class Composition(Transformation):
    def __init__(self, transformations, always_apply, p):
        """
            :args:
                transformations: list of Transformation
                    list of transformations to compose
                always_apply: bool
                    the flag of force application
                p: float
                    the probability of application
        """
        super(Composition, self).__init__(always_apply, p)

        if not isinstance(transformations, list):
            raise RuntimeError(
                'transformations is type of {} that is not list'.format(type(transformations))
            )
        elif not all(isinstance(t, Transformation) for t in transformations):
            for idx, t in enumerate(transformations):
                if not isinstance(t, Transformation):
                    raise RuntimeError(
                        'object at {} position is not subtype of Transformation'.format(idx)
                    )

        self.transformations = transformations

    def get_state_dict(self):
        state = super().get_state_dict()

        state.update(self.get_class_fullname_as_dict())
        state.update({'transformations': [t.get_state_dict() for t in self.transformations]})

        return state

    def __len__(self):
        return len(self.transformations)

    def __getitem__(self, idx):
        return self.transformations[idx]

    def keys(self):
        """Return the keys that are transformed by target functions
        """
        return set.union(*(t.keys() for t in self.transformations)) if self.transformations else set()

    def add_keys(self, additional_keys):
        """Add keys to transform them the same way as one of existing targets.

            :NOTE:
                for example: additional_keys = {'voxel2': 'voxel'}
                adds new key "voxel2" that transformed by "voxel" target function

            :args:
                additional_keys: dict
                    keys - additional key name, values - existed target name.
        """
        for t in self.transformations: t.add_keys(additional_keys)

        return self

    def __repr__(self):
        return self.repr()

    def repr(self, indent=Transformation.REPR_INDENT_STEP):
        args = self.get_base_init_args()

        repr_string = self.get_class_name() + '(['

        for t in self.transformations:
            repr_string += '\n'

            if hasattr(t, 'repr'):
                t_repr = t.repr(indent + self.REPR_INDENT_STEP)
            else:
                t_repr = repr(t)

            repr_string += ' ' * indent + t_repr + ','

        repr_string += '\n' + ' ' * (indent - self.REPR_INDENT_STEP) + '], {args})'.format(args=format_args(args))

        return repr_string

@register_as_serializable
class Sequential(Composition):
    """Compose transformations to apply sequentially.
    """
    def __init__(self, transformations, always_apply=False, p=1.0):
        """
            :args:
                transformations: list of Apply
                    list of transformations to apply sequentially
                always_apply: bool
                    the flag of force application
                p: float
                    the probability of application
        """
        super(Sequential, self).__init__(transformations, always_apply, p)

    def __call__(self, *args, force_apply=False, **data):
        if self.whether_apply(force_apply):
            for transform in self.transformations:
                data = transform(**data)

        return data

@register_as_serializable
class NonSequential(Sequential):
    """Compose transformations to apply sequentially in random order.
    """
    def __call__(self, *args, force_apply=False, **data):
        if self.whether_apply(force_apply):
            np.random.shuffle(self.transformations)

            for transform in self.transformations:
                data = transform(**data)

        return data

@register_as_serializable
class OneOf(Composition):
    """Select one of transformations to apply.
    """
    def __init__(self, transformations, always_apply=False, p=0.5):
        """
            :NOTE:
                transform probabilities will be normalized to one 1, so in this case transformations probabilities works as weights.

            :args:
                transformations: list of Apply
                    list of transformations to select one to apply
                always_apply: bool
                    the flag of force application
                p: float
                    the probability of application
        """
        super(OneOf, self).__init__(transformations, always_apply, p)

        transformations_ps = [t.p for t in self.transformations]
        s = sum(transformations_ps)

        self.transformations_ps = [t / s for t in transformations_ps]

    def __call__(self, *args, force_apply = False, **data):
        if self.transformations_ps and self.whether_apply(force_apply):
            transform = np.random.choice(self.transformations, p=self.transformations_ps)
            data = transform(force_apply=True, **data)

        return data
