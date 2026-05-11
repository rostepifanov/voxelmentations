import numpy as np

from functools import wraps

import voxelmentations.core.constants as C

def contiguous(func):
    """Ensure that the function output is C-contiguous."""

    @wraps(func)
    def wfunc(*args, **kwargs):
        output = func(*args, **kwargs)

        return np.require(output, requirements=['C_CONTIGUOUS'])

    return wfunc

def preserve_channel_dim(func):
    """Preserve dummy channel dim."""

    @wraps(func)
    def wfunc(input, *args, **kwargs):
        shape = input.shape
        output = func(input, *args, **kwargs)

        if len(shape) == C.NUM_MULTI_CHANNEL_DIMENSIONS and shape[C.CHANNEL_DIM] == 1 and output.ndim == C.NUM_MONO_CHANNEL_DIMENSIONS:
            return np.expand_dims(output, axis=C.CHANNEL_DIM)

        if len(shape) == C.NUM_MONO_CHANNEL_DIMENSIONS and output.ndim == C.NUM_MULTI_CHANNEL_DIMENSIONS:
            return output[:, :, 0]

        return output

    return wfunc
