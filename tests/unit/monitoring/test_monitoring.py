from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from nonconform import ConformalDetector, Split
from nonconform.martingales import PowerMartingale
from nonconform.monitoring import (
    ExchangeabilityMonitor,
    SequentialRankConformalizer,
)
from nonconform.resampling import CrossValidation


class DistanceFromMeanDetector(BaseEstimator):
    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> DistanceFromMeanDetector:
        _ = y
        self.center_ = np.mean(X, axis=0)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.linalg.norm(X - self.center_, axis=1)


class TestSequentialRankConformalizer:
    def test_upper_tail_randomized_ranks_match_definition(self):
        seed = 7
        uniforms = np.random.default_rng(seed).random(3)
        conformalizer = SequentialRankConformalizer(seed=seed)

        actual = conformalizer.update_many(np.array([2.0, 1.0, 2.0]))

        expected = np.array(
            [
                uniforms[0],
                (1.0 + uniforms[1]) / 2.0,
                2.0 * uniforms[2] / 3.0,
            ]
        )
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_array_equal(conformalizer.scores, [1.0, 2.0, 2.0])

    def test_lower_tail_randomized_ranks_match_definition(self):
        seed = 11
        uniforms = np.random.default_rng(seed).random(3)
        conformalizer = SequentialRankConformalizer(tail="lower", seed=seed)

        actual = conformalizer.update_many(np.array([2.0, 1.0, 2.0]))

        expected = np.array(
            [uniforms[0], uniforms[1] / 2.0, (1.0 + 2.0 * uniforms[2]) / 3.0]
        )
        np.testing.assert_allclose(actual, expected)

    def test_prime_establishes_history_without_consuming_rng(self):
        seed = 19
        expected_uniform = float(np.random.default_rng(seed).random())
        conformalizer = SequentialRankConformalizer(seed=seed)

        returned = conformalizer.prime_many(np.array([0.0, 1.0]))
        p_value = conformalizer.update(0.5)

        assert returned is conformalizer
        assert p_value == pytest.approx((1.0 + expected_uniform) / 3.0)
        assert conformalizer.count == 3

    def test_reset_restores_history_and_rng(self):
        conformalizer = SequentialRankConformalizer(seed=23)
        first = conformalizer.update_many(np.array([0.1, 0.1, 0.8]))

        conformalizer.reset()
        second = conformalizer.update_many(np.array([0.1, 0.1, 0.8]))

        np.testing.assert_allclose(first, second)
        assert conformalizer.count == 3

    @pytest.mark.parametrize("tail", ["middle", "", None])
    def test_invalid_tail_raises(self, tail):
        with pytest.raises(ValueError, match="tail"):
            SequentialRankConformalizer(tail=tail)  # type: ignore[arg-type]

    @pytest.mark.parametrize("seed", [-1, 1.5, True])
    def test_invalid_seed_raises(self, seed):
        with pytest.raises(ValueError, match="seed"):
            SequentialRankConformalizer(seed=seed)  # type: ignore[arg-type]

    @pytest.mark.parametrize("score", [np.nan, np.inf, -np.inf, "bad"])
    def test_invalid_score_raises(self, score):
        with pytest.raises(ValueError, match="score"):
            SequentialRankConformalizer().update(score)  # type: ignore[arg-type]

    def test_multidimensional_score_input_raises(self):
        conformalizer = SequentialRankConformalizer()
        with pytest.raises(ValueError, match="one-dimensional"):
            conformalizer.update_many(np.ones((2, 1)))
        with pytest.raises(ValueError, match="one-dimensional"):
            conformalizer.prime_many(np.ones((2, 1)))


class TestExchangeabilityMonitor:
    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(42)
        return (
            rng.normal(size=(30, 3)),
            rng.normal(size=(8, 3)),
            rng.normal(size=(6, 3)),
        )

    def test_fit_prime_update_exposes_complete_state(self, data):
        x_train, x_reference, x_stream = data
        monitor = ExchangeabilityMonitor(
            DistanceFromMeanDetector(),
            martingale=PowerMartingale(epsilon=0.5),
            score_polarity="higher_is_anomalous",
            seed=17,
        ).fit(x_train)

        returned = monitor.prime(x_reference)
        state = monitor.update(x_stream[0])

        assert returned is monitor
        assert state.rank_step == len(x_reference) + 1
        assert state.evidence_step == 1
        assert 0.0 <= state.p_value <= 1.0
        assert state.e_value == state.martingale_state.e_value
        assert state.log_e_value == state.martingale_state.log_e_value
        assert state.martingale == state.martingale_state.martingale
        assert state.restarted_martingale == state.martingale_state.restarted_martingale
        assert monitor.state is state

    def test_update_many_matches_iterative_updates(self, data):
        x_train, x_reference, x_stream = data

        def make_monitor():
            return (
                ExchangeabilityMonitor(
                    DistanceFromMeanDetector(),
                    martingale=PowerMartingale(epsilon=0.7),
                    score_polarity="higher_is_anomalous",
                    seed=29,
                )
                .fit(x_train)
                .prime(x_reference)
            )

        many = make_monitor().update_many(x_stream)
        iterative_monitor = make_monitor()
        iterative = [iterative_monitor.update(row) for row in x_stream]

        np.testing.assert_allclose(
            [state.p_value for state in many],
            [state.p_value for state in iterative],
        )
        np.testing.assert_allclose(
            [state.martingale for state in many],
            [state.martingale for state in iterative],
        )

    def test_prime_after_monitoring_starts_raises(self, data):
        x_train, x_reference, x_stream = data
        monitor = ExchangeabilityMonitor(
            DistanceFromMeanDetector(),
            score_polarity="higher_is_anomalous",
        ).fit(x_train)
        monitor.update(x_stream[0])

        with pytest.raises(RuntimeError, match="after evidence monitoring starts"):
            monitor.prime(x_reference)

    def test_unfitted_monitor_rejects_prime_and_update(self, data):
        _, x_reference, x_stream = data
        monitor = ExchangeabilityMonitor(
            DistanceFromMeanDetector(),
            score_polarity="higher_is_anomalous",
        )

        with pytest.raises(NotFittedError):
            monitor.prime(x_reference)
        with pytest.raises(NotFittedError):
            monitor.update(x_stream[0])

    def test_reset_retains_fitted_scorer_and_resets_sequential_state(self, data):
        x_train, x_reference, x_stream = data
        monitor = ExchangeabilityMonitor(
            DistanceFromMeanDetector(),
            score_polarity="higher_is_anomalous",
            seed=31,
        ).fit(x_train)
        monitor.prime(x_reference)
        first = monitor.update(x_stream[0])

        monitor.reset()
        monitor.prime(x_reference)
        second = monitor.update(x_stream[0])

        assert monitor.is_fitted
        assert monitor.state is second
        assert first.p_value == pytest.approx(second.p_value)
        assert first.score == pytest.approx(second.score)

    def test_feature_validation(self, data):
        x_train, _, x_stream = data
        monitor = ExchangeabilityMonitor(
            DistanceFromMeanDetector(),
            score_polarity="higher_is_anomalous",
        ).fit(x_train)

        with pytest.raises(ValueError, match="one-dimensional"):
            monitor.update(x_stream[:2])
        with pytest.raises(ValueError, match="expects 3"):
            monitor.update(np.ones(2))
        with pytest.raises(ValueError, match="two-dimensional"):
            monitor.prime(np.ones(3))
        with pytest.raises(ValueError, match="finite"):
            monitor.update(np.array([0.0, np.nan, 0.0]))

    def test_nonnumeric_feature_inputs_raise_value_error(self, data):
        x_train, _, _ = data
        monitor = ExchangeabilityMonitor(
            DistanceFromMeanDetector(),
            score_polarity="higher_is_anomalous",
        ).fit(x_train)

        with pytest.raises(ValueError, match="finite numeric feature vector"):
            monitor.update(np.array(["bad", "input", "values"], dtype=object))
        with pytest.raises(ValueError, match="finite numeric feature matrix"):
            monitor.update_many(np.array([["bad", "input", "values"]], dtype=object))

    def test_nonnumeric_training_input_raises_value_error(self):
        monitor = ExchangeabilityMonitor(
            DistanceFromMeanDetector(),
            score_polarity="higher_is_anomalous",
        )

        with pytest.raises(ValueError, match="finite numeric feature matrix"):
            monitor.fit(np.array([["bad", "input"]], dtype=object))

    def test_rejects_nonempty_components(self):
        conformalizer = SequentialRankConformalizer().prime(1.0)
        with pytest.raises(ValueError, match="empty history"):
            ExchangeabilityMonitor(
                DistanceFromMeanDetector(),
                conformalizer=conformalizer,
                score_polarity="higher_is_anomalous",
            )


class TestSplitDetectorBridge:
    @pytest.fixture
    def fitted_split_detector(self):
        rng = np.random.default_rng(73)
        x_train = rng.normal(size=(40, 4))
        detector = ConformalDetector(
            detector=DistanceFromMeanDetector(),
            strategy=Split(n_calib=10),
            score_polarity="higher_is_anomalous",
            seed=5,
        ).fit(x_train)
        return detector, rng.normal(size=(5, 4))

    def test_bridge_primes_calibration_scores_and_preserves_detector(
        self, fitted_split_detector
    ):
        detector, x_stream = fitted_split_detector
        calibration_before = detector.calibration_set

        monitor = ExchangeabilityMonitor.from_split_detector(
            detector,
            martingale=PowerMartingale(epsilon=0.5),
            seed=41,
        )
        state = monitor.update(x_stream[0])

        assert monitor.is_fitted
        assert state.rank_step == len(calibration_before) + 1
        assert state.evidence_step == 1
        np.testing.assert_array_equal(detector.calibration_set, calibration_before)

    def test_paper_fixed_split_code_remains_executable(self, fitted_split_detector):
        detector, x_stream = fitted_split_detector
        martingale = PowerMartingale(epsilon=0.5)

        for x_t in x_stream:
            p_t = detector.compute_p_value(x_t)
            state = martingale.update(p_t)

        assert state.step == len(x_stream)

    def test_bridge_requires_fitted_unweighted_split_detector(self):
        base = DistanceFromMeanDetector()
        unfitted = ConformalDetector(
            detector=base,
            strategy=Split(n_calib=2),
            score_polarity="higher_is_anomalous",
        )
        with pytest.raises(NotFittedError):
            ExchangeabilityMonitor.from_split_detector(unfitted)

        rng = np.random.default_rng(3)
        cross_validated = ConformalDetector(
            detector=base,
            strategy=CrossValidation(k=2),
            score_polarity="higher_is_anomalous",
        ).fit(rng.normal(size=(10, 2)))
        with pytest.raises(ValueError, match="Split"):
            ExchangeabilityMonitor.from_split_detector(cross_validated)
