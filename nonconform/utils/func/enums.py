from enum import Enum, auto


class Distribution(Enum):
    """Probability distributions for validation set sizes in randomized strategies.

    This enumeration defines the available distribution types for selecting
    validation set sizes in randomized leave-p-out conformal prediction
    strategies.

    Attributes
    ----------
        BETA_BINOMIAL: Beta-binomial distribution for drawing validation fractions.
            Allows tunable mean and variance through alpha/beta parameters.
        UNIFORM: Discrete uniform distribution over a specified range.
            Simple and controlled selection within [p_min, p_max].
        GRID: Discrete distribution over a specified set of values.
            Targeted control with custom probabilities for each p value.
    """

    BETA_BINOMIAL = auto()
    UNIFORM = auto()
    GRID = auto()


class Aggregation(Enum):
    """Aggregation functions for combining multiple model outputs or scores.

    This enumeration lists strategies for aggregating data, commonly employed
    in ensemble methods to combine predictions or scores from several models.

    Attributes
    ----------
        MEAN: Represents aggregation by calculating the arithmetic mean.
            The underlying value is typically ``"mean"``.
        MEDIAN: Represents aggregation by calculating the median.
            The underlying value is typically ``"median"``.
        MINIMUM: Represents aggregation by selecting the minimum value.
            The underlying value is typically ``"minimum"``.
        MAXIMUM: Represents aggregation by selecting the maximum value.
            The underlying value is typically ``"maximum"``.
    """

    MEAN = "mean"
    MEDIAN = "median"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
