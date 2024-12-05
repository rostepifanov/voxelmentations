import numpy as np

import voxelmentations.core.constants as C

from voxelmentations.core.serializable import Serializable
from voxelmentations.core.utils import get_shortest_class_fullname

class Transformation(Serializable):
    """Root class for single and compound transformations
    """

    REPR_INDENT_STEP=2

    def __init__(self, always_apply, p):
        """
            :args:
                always_apply: bool
                    the flag of force application
                p: float
                    the probability of application
        """
        self.always_apply = always_apply
        self.p = p

    def get_base_init_args(self):
        """
            :return:
                output: dict
                    initialization parameters
        """
        return {'always_apply': self.always_apply, 'p': self.p}

    def get_state_dict(self):
        """
            :return:
                output: dict
                    dict of parameters for initialization
        """
        state = super().get_state_dict()
        state.update(self.get_base_init_args())

        return state

    def whether_apply(self, force_apply):
        return force_apply or self.always_apply or (np.random.random() < self.p)

    def __call__(self, *args, force_apply=False, **data):
        raise NotImplementedError

    def get_class_name(self):
        """
            :return:
                output: str
                    the name of class
        """
        return self.__class__.__name__

    @classmethod
    def get_class_fullname(cls):
        """
            :return:
                output: str
                    the full name of class
        """
        return get_shortest_class_fullname(cls)

    @classmethod
    def get_class_fullname_as_dict(cls):
        """
            :return:
                output: dict
                    the full name of class
        """
        return {C.KW_CLASS_FULLNAME: cls.get_class_fullname()}
