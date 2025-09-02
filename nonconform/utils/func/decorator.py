import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd


def _convert_to_numpy(value: Any) -> Any:
    """Convert a value to numpy array if it's a list.

    Args:
        value: The value to potentially convert.

    Returns:
        The converted value or original value if no conversion needed.
    """
    if isinstance(value, list):
        try:
            return np.array(value)
        except ValueError:
            # Handle nested lists by converting each individually
            return [np.array(v) if isinstance(v, list) else v for v in value]
    return value


def _convert_from_numpy(result: Any) -> Any:
    """Convert numpy arrays in result back to lists recursively.

    Args:
        result: The result to convert.

    Returns:
        The converted result with numpy arrays as lists.
    """
    if isinstance(result, np.ndarray):
        return result.tolist()
    elif isinstance(result, tuple):
        return tuple(_convert_from_numpy(x) for x in result)
    elif isinstance(result, list):
        return [_convert_from_numpy(x) for x in result]
    return result


def _performance_conversion(*arg_names: str) -> Callable:
    """Create a decorator to convert specified arguments and return values.

    **Internal use only.** This decorator factory produces a decorator that
    converts specified input arguments from Python lists to numpy arrays before function
    calls and converts numpy arrays in return values back to Python lists.

    Args:
        *arg_names: One or more names of the arguments in the decorated function
            that should be converted from lists to numpy.ndarray.

    Returns:
        The actual decorator that can be applied to a function.

    Note:
        This is an internal utility function. Argument conversion applies to both
        positional and keyword arguments. If a list cannot be directly converted
        to numpy array due to heterogeneous data, it attempts to convert nested
        lists individually.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Convert specified keyword arguments
            converted_kwargs = {
                k: _convert_to_numpy(v) if k in arg_names else v
                for k, v in kwargs.items()
            }

            # Convert specified positional arguments using function signature
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            converted_args = []

            for i, arg_value in enumerate(args):
                should_convert = i < len(param_names) and param_names[i] in arg_names
                converted_args.append(
                    _convert_to_numpy(arg_value) if should_convert else arg_value
                )

            # Execute function and convert result
            result = func(*converted_args, **converted_kwargs)
            return _convert_from_numpy(result)

        return wrapper

    return decorator


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
    def wrapper(self, x: pd.DataFrame | np.ndarray, *args, **kwargs) -> Any:
        # Convert pandas.DataFrame to numpy.ndarray if necessary
        if isinstance(x, pd.DataFrame):
            x_converted = x.values
        else:
            x_converted = x
        return func(self, x_converted, *args, **kwargs)

    return wrapper
