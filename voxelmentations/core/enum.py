import enum

class BorderType(enum.Enum):
    CONSTANT = 'constant'
    REPLICATE = 'replicate'
    REFLECT_1001 = 'reflect'
    REFLECT_101 = 'reflect'
    WRAP = 'wrap'
    DEFAULT = CONSTANT

class InterType(enum.Enum):
    NEAREST = 'nearest'
    LINEAR = 'linear'
    DEFAULT = LINEAR
    MASK_DEFAULT = NEAREST

class PositionType(enum.Enum):
    CENTER = 'center'
    LEFT = 'left'
    RIGHT = 'right'
    RANDOM = 'random'
