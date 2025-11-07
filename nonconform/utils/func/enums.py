from enum import Enum, auto


class Distribution(Enum):
    """Probability distributions for validation set sizes in randomized strategies.

    This enumeration defines the available distribution types for selecting
    validation set sizes in randomized leave-p-out conformal prediction
    strategies.

    Attributes:
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

    Attributes:
        MEAN: Represents aggregation by calculating the arithmetic mean.
            The underlying value is typically ``"mean"``.
        MEDIAN: Represents aggregation by calculating the median.
            The underlying value is typically ``"median"``.
        MINIMUM: Represents aggregation by selecting the minimum value.
            The underlying value is typically ``"minimum"``.
        MAXIMUM: Represents aggregation by selecting the maximum value.
            The underlying value is typically ``"maximum"``.
    """

    MEAN = auto()
    MEDIAN = auto()
    MINIMUM = auto()
    MAXIMUM = auto()


class Dataset(Enum):
    """Available datasets for anomaly detection experiments.

    This enumeration provides all built-in datasets that can be loaded
    using the load() function. Each dataset is preprocessed for anomaly
    detection tasks with normal and anomalous samples.

    Usage:
        from nonconform.utils.data import load, Dataset
        df = load(Dataset.FRAUD, setup=True, seed=42)
    """

    ANNTHYROID = "annthyroid"
    BACKDOOR = "backdoor"
    BREAST = "breast"
    CARDIO = "cardio"
    COVER = "cover"
    DONORS = "donors"
    FRAUD = "fraud"
    GLASS = "glass"
    HEPATITIS = "hepatitis"
    HTTP = "http"
    IONOSPHERE = "ionosphere"
    LETTER = "letter"
    LYMPHOGRAPHY = "lymphography"
    MAGIC_GAMMA = "magic_gamma"
    MAMMOGRAPHY = "mammography"
    MNIST = "mnist"
    MUSK = "musk"
    OPTDIGITS = "optdigits"
    PAGEBLOCKS = "pageblocks"
    PENDIGITS = "pendigits"
    SATIMAGE2 = "satimage2"
    SHUTTLE = "shuttle"
    SMTP = "smtp"
    STAMPS = "stamps"
    THYROID = "thyroid"
    VOWELS = "vowels"
    WBC = "wbc"
    WINE = "wine"
    YEAST = "yeast"


class Pruning(Enum):
    """Available pruning strategies for weighted FDR control.

    This enumeration lists all strategies available for controlling
    the FDR under potential covariate shift.

    Attributes:
        HETEROGENEOUS: Heterogeneous pruning removes elements
            based on independent random checks per item.
        HOMOGENEOUS: Homogeneous pruning applies one shared
            random decision to all items.
        DETERMINISTIC: Deterministic pruning removes items
            using a fixed rule with no randomness.
    """

    HETEROGENEOUS = auto()
    HOMOGENEOUS = auto()
    DETERMINISTIC = auto()


class Kernel(Enum):
    """Available kernels for the smoothed p-value computation."""

    GAUSSIAN = "gaussian"
    EXPONENTIAL = "exponential"
    BOX = "box"
    TRIANGULAR = "tri"
    EPANECHNIKOV = "epa"
    BIWEIGHT = "biweight"
    TRIWEIGHT = "triweight"
    TRICUBE = "tricube"
    COSINE = "cosine"
