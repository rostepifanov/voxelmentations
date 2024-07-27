import enum

class BorderType(enum.Enum):
    CONSTANT = 'constant'
    REPLICATE = 'replicate'

class InterType(enum.Enum):
    NEAREST = 'nearest'
    LINEAR = 'linear'

class PositionType(enum.Enum):
    CENTER = 'center'
    LEFT = 'left'
    RIGHT = 'right'
    RANDOM = 'random'
