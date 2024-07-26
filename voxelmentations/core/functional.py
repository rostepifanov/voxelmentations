import numpy as np

from functools import wraps

import voxelmentations.core.constants as C

def preserve_channel_dim(func):
    """Preserve dummy channel dim."""

    @wraps(func)
    def wfunc(input, *args, **kwargs):
        shape = input.shape
        output = func(input, *args, **kwargs)

        if len(shape) == C.NUM_MULTI_CHANNEL_DIMENSIONS and shape[-1] == 1 and output.ndim == C.NUM_MONO_CHANNEL_DIMENSIONS:
            return np.expand_dims(output, axis=-1)

        if len(shape) == C.NUM_MONO_CHANNEL_DIMENSIONS and output.ndim == C.NUM_MULTI_CHANNEL_DIMENSIONS:
            return output[:, :, 0]

        return output

    return wfunc
