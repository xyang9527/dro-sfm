
import numpy as np
import logging
import torch.nn as nn
from dro_sfm.utils.types import is_list

########################################################################################################################

class ProgressiveScaling:
    """
    Helper class to manage progressive scaling.
    After a certain training progress percentage, decrease the number of scales by 1.

    Parameters
    ----------
    progressive_scaling : float
        Training progress percentage where the number of scales is decreased
    num_scales : int
        Initial number of scales
    """
    def __init__(self, progressive_scaling, num_scales=4):
        logging.warning(f'ProgressiveScaling::__init__(progressive_scaling={progressive_scaling}, num_scales={num_scales})')
        self.num_scales = num_scales
        # Use it only if bigger than zero (make a list)
        if progressive_scaling > 0.0:
            self.progressive_scaling = np.float32(
                [progressive_scaling * (i + 1) for i in range(num_scales - 1)] + [1.0])
        # Otherwise, disable it
        else:
            self.progressive_scaling = progressive_scaling
    def __call__(self, progress):
        """
        Call for an update in the number of scales

        Parameters
        ----------
        progress : float
            Training progress percentage

        Returns
        -------
        num_scales : int
            New number of scales
        """
        if is_list(self.progressive_scaling):
            return int(self.num_scales -
                       np.searchsorted(self.progressive_scaling, progress))
        else:
            return self.num_scales

########################################################################################################################

class LossBase(nn.Module):
    """Base class for losses."""
    def __init__(self):
        """Initializes logs and metrics dictionaries"""
        logging.warning(f'LossBase::__init__()')
        super().__init__()
        self._logs = {}
        self._metrics = {}

########################################################################################################################

    @property
    def logs(self):
        """Return logs."""
        return self._logs

    @property
    def metrics(self):
        """Return metrics."""
        return self._metrics

    def add_metric(self, key, val):
        """Add a new metric to the dictionary and detach it."""
        self._metrics[key] = val.detach()

########################################################################################################################
