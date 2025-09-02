import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd


def _performance_conversion(*arg_names: str) -> Callable:
    """Create a decorator to convert specified arguments and return values.

    **Internal use only.** This decorator factory produces a decorator that
    converts specified input arguments from Python lists to numpy arrays before function
    calls and converts numpy arrays in return values back to Python lists.

    Args:
        *arg_names: One or more names of the arguments in the decorated function
            that should be converted from lists to numpy.ndarray.

    Returns
    -------
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
            # Internal helper to convert values to numpy arrays
            def convert_to_array(value: Any) -> Any:
                if isinstance(value, list):
                    try:
                        return np.array(value)
                    except ValueError:
                        # Attempt to convert nested lists if direct conversion fails
                        return [
                            np.array(v) if isinstance(v, list) else v for v in value
                        ]
                return value

            # Convert specified keyword arguments
            new_kwargs = {
                k: convert_to_array(v) if k in arg_names else v
                for k, v in kwargs.items()
            }

            # Convert specified positional arguments
            # Use inspect module for more robust parameter inspection
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            new_args = []
            for i, arg_val in enumerate(args):
                if i < len(param_names) and param_names[i] in arg_names:
                    new_args.append(convert_to_array(arg_val))
                else:
                    new_args.append(arg_val)
            args = tuple(new_args)

            result = func(*args, **new_kwargs)

            # Internal helper to convert numpy arrays in results back to lists
            def convert_result(r: Any) -> Any:
                if isinstance(r, np.ndarray):
                    return r.tolist()
                elif isinstance(r, tuple):
                    return tuple(convert_result(x) for x in r)
                elif isinstance(r, list):
                    return [convert_result(x) for x in r]
                return r

            return convert_result(result)

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

    Returns
    -------
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
