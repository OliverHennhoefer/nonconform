"""Enumeration constants for nonconform.

This module provides enum classes used throughout the package.
"""

from enum import Enum, auto


class Distribution(Enum):
    """Probability distributions for validation set sizes in randomized strategies.

    Attributes:
        BETA_BINOMIAL: Beta-binomial distribution for drawing validation fractions.
        UNIFORM: Discrete uniform distribution over a specified range.
        GRID: Discrete distribution over a specified set of values.
    """

    BETA_BINOMIAL = auto()
    UNIFORM = auto()
    GRID = auto()


class Aggregation(Enum):
    """Aggregation functions for combining multiple model outputs or scores.

    Attributes:
        MEAN: Aggregate by calculating the arithmetic mean.
        MEDIAN: Aggregate by calculating the median.
        MINIMUM: Aggregate by selecting the minimum value.
        MAXIMUM: Aggregate by selecting the maximum value.
    """

    MEAN = auto()
    MEDIAN = auto()
    MINIMUM = auto()
    MAXIMUM = auto()


class Pruning(Enum):
    """Pruning strategies for weighted FDR control.

    Attributes:
        HETEROGENEOUS: Remove elements based on independent random checks per item.
        HOMOGENEOUS: Apply one shared random decision to all items.
        DETERMINISTIC: Remove items using a fixed rule with no randomness.
    """

    HETEROGENEOUS = auto()
    HOMOGENEOUS = auto()
    DETERMINISTIC = auto()


class Kernel(Enum):
    """Kernel functions for KDE-based p-value computation.

    Attributes:
        GAUSSIAN: Gaussian (normal) kernel.
        EXPONENTIAL: Exponential kernel.
        BOX: Box (uniform) kernel.
        TRIANGULAR: Triangular kernel.
        EPANECHNIKOV: Epanechnikov kernel.
        BIWEIGHT: Biweight (quartic) kernel.
        TRIWEIGHT: Triweight kernel.
        TRICUBE: Tricube kernel.
        COSINE: Cosine kernel.
    """

    GAUSSIAN = "gaussian"
    EXPONENTIAL = "exponential"
    BOX = "box"
    TRIANGULAR = "tri"
    EPANECHNIKOV = "epa"
    BIWEIGHT = "biweight"
    TRIWEIGHT = "triweight"
    TRICUBE = "tricube"
    COSINE = "cosine"


__all__ = [
    "Aggregation",
    "Distribution",
    "Kernel",
    "Pruning",
]
