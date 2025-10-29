from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd


def _ensure_numpy_array(func: Callable) -> Callable:
    """Ensure a specific input argument is a numpy array.

    **Internal use only.** This decorator is designed for methods where the first
    argument after `self` (conventionally named `x`) is expected to be a numpy array.
    Automatically converts pandas DataFrame to numpy array using .values attribute.

    Args:
        func: The method to be decorated. Must have `self` as first parameter,
            followed by the data argument `x`.

    Returns:
        The wrapped method that will receive `x` as a numpy array.

    Note:
        This is an internal utility decorator used throughout the package to ensure
        consistent data types for detector methods.
    """

    @wraps(func)
    def wrapper(self, x: pd.DataFrame | pd.Series | np.ndarray, *args, **kwargs) -> Any:
        # Convert pandas objects without forcing a copy
        if isinstance(x, (pd.DataFrame | pd.Series)):
            x_converted = x.to_numpy(copy=False)
        else:
            x_converted = x
        return func(self, x_converted, *args, **kwargs)

    return wrapper
