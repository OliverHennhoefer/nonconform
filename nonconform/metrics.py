"""Public helpers for score aggregation and labeled evaluation.

``false_discovery_rate`` retains its v1 public name but returns the realized
false discovery proportion for one supplied testing family. ``statistical_power``
similarly returns the realized true positive rate. Expected FDR and statistical
power are repeated-sampling properties, not quantities identified by one labeled
family.
"""

from nonconform._internal.math_utils import (
    aggregate,
    false_discovery_rate,
    statistical_power,
)

__all__ = [
    "aggregate",
    "false_discovery_rate",
    "statistical_power",
]
