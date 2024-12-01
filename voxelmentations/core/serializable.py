from voxelmentations import __version__

REGISTRY_SERIALIZABLE = {}

def register_as_serializable(cls):
    REGISTRY_SERIALIZABLE[cls.get_class_fullname()] = cls

    return cls

class Serializable(object):
    """Class to serialize transforms
    """
    def get_state_dict(self):
        return {}

    def to_dict(self):
        """Convert a pipeline to dict repr that uses only standard python data types
        """
        state_dict = self.get_state_dict()

        return {'__version__': __version__, 'transformation': state_dict}

def from_dict(state_dict):
    transformation = state_dict['transformation']

    name = transformation.pop('__class_fullname__')
    cls = REGISTRY_SERIALIZABLE[name]

    args = {}

    for key, value in transformation.items():
        if key == 'transformations':
            args['transformations'] = [
                from_dict({'transformation': t}) for t in transformation['transformations']
            ]
        else:
            args[key] = value

    return cls(**args)
