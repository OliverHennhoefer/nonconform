import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from nonconform.estimation.base import BaseConformalDetector
from nonconform.strategy.base import BaseStrategy
from nonconform.utils.func.decorator import ensure_numpy_array
from nonconform.utils.func.enums import Aggregation
from nonconform.utils.func.logger import get_logger
from nonconform.utils.func.params import set_params
from nonconform.utils.stat.aggregation import aggregate
from nonconform.utils.stat.statistical import calculate_weighted_p_val
from pyod.models.base import BaseDetector


class WeightedConformalDetector(BaseConformalDetector):
    """Weighted conformal anomaly detector with covariate shift adaptation.

    This detector inherits from BaseConformalDetector and implements a conformal
    prediction framework for anomaly detection, incorporating weights to adapt to
    potential covariate shifts between calibration and test data. It leverages an
    underlying PyOD detector and a calibration strategy.

    The weighting mechanism estimates the density ratio between calibration
    and test instances using a logistic regression model trained to
    distinguish between them. These weights are then used in the calculation
    of p-values.

    The methodology is inspired by concepts for handling covariate shift in
    conformal prediction, adapted for anomaly detection.

    Attributes
    ----------
        detector (BaseDetector): The underlying PyOD anomaly detection model,
            initialized with the specified seed.
        strategy (BaseStrategy): The calibration strategy (e.g., Bootstrap,
            CrossValidation) used to generate calibration scores and identify
            calibration samples.
        aggregation (Aggregation): Method used for aggregating scores from
            multiple detector models.
        seed (int): Random seed for reproducibility in stochastic processes.
        detector_set (List[BaseDetector]): A list of one or more trained
            detector instances, populated by the `fit` method via the strategy.
        calibration_set (List[float]): A list of non-conformity scores obtained
            from the calibration process, populated by the `fit` method.
        calibration_samples (numpy.ndarray): The actual data instances from the
            input `x` that were used for calibration, identified by the
            strategy. Populated by the `fit` method.
    """

    def __init__(
        self,
        detector: BaseDetector,
        strategy: BaseStrategy,
        aggregation: Aggregation = Aggregation.MEDIAN,
        seed: int | None = None,
    ):
        """Initialize the WeightedConformalDetector.

        Args:
            detector (BaseDetector): A PyOD anomaly detector instance. It will
                be configured with the specified seed.
            strategy (BaseStrategy): A calibration strategy instance.
            aggregation (Aggregation, optional): Method used for aggregating
                scores from multiple detector models. Defaults to Aggregation.MEDIAN.
            seed (int | None, optional): Random seed for reproducibility.
                Defaults to None.

        Raises
        ------
            ValueError: If seed is negative.
            TypeError: If aggregation is not an Aggregation enum.
        """
        if seed is not None and seed < 0:
            raise ValueError(f"seed must be a non-negative integer or None, got {seed}")
        if not isinstance(aggregation, Aggregation):
            raise TypeError(
                f"aggregation must be an Aggregation enum, got {type(aggregation)}"
            )

        self.detector: BaseDetector = set_params(detector, seed)
        self.strategy: BaseStrategy = strategy
        self.aggregation: Aggregation = aggregation
        self.seed: int | None = seed

        self.detector_set: list[BaseDetector] = []
        self.calibration_set: list[float] = []
        self.calibration_samples: np.ndarray = np.array([])  # Initialize as empty

    @ensure_numpy_array
    def fit(self, x: pd.DataFrame | np.ndarray, iteration_callback=None) -> None:
        """Fits the detector and prepares for conformal prediction.

        This method uses the provided strategy to fit the underlying detector(s)
        and generate a set of calibration scores. It also identifies and stores
        the data samples used for calibration. The `weighted` flag is passed
        as ``True`` to the strategy's `fit_calibrate` method, signaling that
        calibration sample identification is required.

        Args:
            x (Union[pandas.DataFrame, numpy.ndarray]): The input data used for
                training/fitting the detector(s) and for calibration. The
                `@ensure_numpy_array` decorator converts `x` to a
                ``numpy.ndarray`` internally.
            iteration_callback (callable, optional): Optional callback function
                for strategies that support iteration tracking. Defaults to None.
        """
        self.detector_set, self.calibration_set = self.strategy.fit_calibrate(
            x=x,
            detector=self.detector,
            weighted=True,
            seed=self.seed,
            iteration_callback=iteration_callback,
        )
        if (
            self.strategy.calibration_ids is not None
            and len(self.strategy.calibration_ids) > 0
        ):
            self.calibration_samples = x[self.strategy.calibration_ids]
        else:
            # Handle case where calibration_ids might be empty or None
            # This might happen if the strategy doesn't yield IDs or x is too small
            self.calibration_samples = np.array([])

    @ensure_numpy_array
    def predict(
        self,
        x: pd.DataFrame | np.ndarray,
        raw: bool = False,
    ) -> np.ndarray:
        """Generate weighted anomaly estimates (p-values or raw scores) for new data.

        For each test instance in `x`:
        1. Anomaly scores are obtained from each detector in `detector_set`.
        2. These scores are aggregated using the method specified in `self.aggregation`.
        3. Importance weights are computed for calibration and test instances
           to account for covariate shift, using `_compute_weights`.
        4. Based on the `raw` parameter, either returns the aggregated scores
           or weighted p-values calculated using the aggregated scores,
           calibration scores, and computed weights.

        Args:
            x (Union[pandas.DataFrame, numpy.ndarray]): The input data for which
                anomaly estimates are to be generated. The `@ensure_numpy_array`
                decorator converts `x` to a ``numpy.ndarray`` internally.
            raw (bool, optional): Whether to return raw anomaly scores or
                weighted p-values. Defaults to False.
                * If True: Returns the aggregated anomaly scores from the
                  detector set for each data point.
                * If False: Returns the weighted p-values for each data point,
                  accounting for covariate shift between calibration and test data.

        Returns
        -------
            numpy.ndarray: An array containing the anomaly estimates. The content of the
            array depends on the `raw` argument:
            - If raw=True, an array of anomaly scores (float).
            - If raw=False, an array of weighted p-values (float).
        """
        logger = get_logger("estimation.weighted_conformal")
        scores_list = [
            model.decision_function(x)
            for model in tqdm(
                self.detector_set,
                total=len(self.detector_set),
                desc=f"Aggregating {len(self.detector_set)} models",
                disable=not logger.isEnabledFor(logging.INFO),
            )
        ]

        w_cal, w_x = self._compute_weights(x)
        estimates = aggregate(self.aggregation, np.array(scores_list))

        return (
            estimates
            if raw
            else calculate_weighted_p_val(
                np.array(estimates),
                np.array(self.calibration_set),
                np.array(w_x),
                np.array(w_cal),
            )
        )

    def _compute_weights(
        self, test_instances: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute importance weights for calibration and test instances.

        This method trains a logistic regression classifier to distinguish
        between samples from the calibration distribution and samples from the
        test distribution. The probabilities from this classifier are used
        to estimate the density ratio w(z) = p_test(z) / p_calib(z).

        The weights are clipped to a predefined range [0.35, 45] to prevent
        extreme values.

        Args:
            test_instances (numpy.ndarray): The test data instances for which
                weights need to be computed.

        Returns
        -------
            Tuple[numpy.ndarray, numpy.ndarray]:
                A tuple containing:
                * clipped_w_calib: Clipped weights for calibration samples.
                * clipped_w_tests: Clipped weights for test instances.
        """
        if self.calibration_samples.shape[0] == 0:
            raise ValueError(
                "Calibration samples are empty. Weights cannot be computed. "
                "Ensure fit() was called and strategy provided calibration_ids."
            )

        calib_labeled = np.hstack(
            (self.calibration_samples, np.zeros((self.calibration_samples.shape[0], 1)))
        )
        tests_labeled = np.hstack(
            (test_instances, np.ones((test_instances.shape[0], 1)))
        )

        joint_labeled = np.vstack((calib_labeled, tests_labeled))
        rng = np.random.default_rng(seed=self.seed)
        rng.shuffle(joint_labeled)

        x_joint = joint_labeled[:, :-1]
        y_joint = joint_labeled[:, -1]

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1_000,
                random_state=self.seed,
                verbose=0,
                class_weight="balanced",
            ),
        )
        model.fit(x_joint, y_joint)

        calib_prob = model.predict_proba(self.calibration_samples)
        tests_prob = model.predict_proba(test_instances)

        # Density ratio w(z) = p_test(z) / p_calib(z)
        # p_calib(z) = P(label=0 | z) ; p_test(z) = P(label=1 | z)
        # For calibration samples, weight is P(label=1 | z_calib) / P(label=0 | z_calib)
        # For test samples, weight is P(label=1 | z_test) / P(label=0 | z_test)
        # These are likelihood ratios p(z | test) / p(z | calib)
        w_calib = calib_prob[:, 1] / (
            calib_prob[:, 0] + 1e-9
        )  # Add epsilon for stability
        w_tests = tests_prob[:, 1] / (
            tests_prob[:, 0] + 1e-9
        )  # Add epsilon for stability

        clipped_w_calib = np.clip(w_calib, 0.35, 45.0)
        clipped_w_tests = np.clip(w_tests, 0.35, 45.0)

        return clipped_w_calib, clipped_w_tests
