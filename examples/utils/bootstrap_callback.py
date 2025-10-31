import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import (
    entropy,
    false_discovery_control,
    gaussian_kde,
    wasserstein_distance,
)

from nonconform.detection import ConformalDetector
from nonconform.strategy import Bootstrap
from nonconform.utils.data import Dataset, load
from nonconform.utils.stat import false_discovery_rate, statistical_power
from pyod.models.iforest import IForest

if __name__ == "__main__":
    x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=1)

    # Track iterations with multiple convergence metrics
    all_scores = []
    emd_distances = []
    kde_js_divergences = []
    kde_kl_divergences = []
    calibration_sizes = []

    def track_iterations(iteration: int, scores):
        all_scores.append(scores.copy())

        # Compute cumulative distribution
        cumulative = np.concatenate(all_scores)
        calibration_sizes.append(len(cumulative))

        # Compute Earth Mover's Distance from previous cumulative distribution
        if iteration > 0:
            prev_cumulative = np.concatenate(all_scores[:-1])
            emd = wasserstein_distance(prev_cumulative, cumulative)
            emd_distances.append(emd)

            # Compute KDE-based metrics (sample-size independent)
            try:
                # Fit KDEs with Scott's rule (automatic bandwidth)
                kde_prev = gaussian_kde(prev_cumulative)
                kde_curr = gaussian_kde(cumulative)

                # Evaluate on shared grid
                score_min = min(prev_cumulative.min(), cumulative.min())
                score_max = max(prev_cumulative.max(), cumulative.max())
                grid = np.linspace(score_min, score_max, 100)

                p = kde_prev(grid)
                q = kde_curr(grid)

                # Normalize to proper probability distributions
                p = p / np.trapezoid(p, grid)
                q = q / np.trapezoid(q, grid)

                # Jensen-Shannon Divergence
                m = 0.5 * (p + q)
                js_div = 0.5 * entropy(p, m) + 0.5 * entropy(q, m)
                kde_js_divergences.append(js_div)

                # KL Divergence (add small epsilon for numerical stability)
                kl_div = entropy(p, q + 1e-10)
                kde_kl_divergences.append(kl_div)

            except Exception:
                # If KDE fails, skip this iteration
                kde_js_divergences.append(np.nan)
                kde_kl_divergences.append(np.nan)

    model = IForest(behaviour="new")
    strategy = Bootstrap(n_calib=5_000, resampling_ratio=0.995)
    ce = ConformalDetector(detector=model, strategy=strategy)
    ce.fit(x_train, iteration_callback=track_iterations)
    estimates = ce.predict(x_test)

    decisions = false_discovery_control(estimates, method="bh") <= 0.2

    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")

    if emd_distances:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        calib_sizes_for_metrics = calibration_sizes[
            1:
        ]  # Skip first iteration (no metrics)

        # Left y-axis: EMD (sample-size dependent)
        emd_safe = np.maximum(
            emd_distances, 1e-12
        )  # Ensure positive values for log scale
        ax1.plot(
            calib_sizes_for_metrics,
            emd_safe,
            "bo-",
            linewidth=2,
            markersize=6,
            label="EMD",
        )
        ax1.set_xlabel("Number of Calibration Samples")
        ax1.set_ylabel("Earth Mover's Distance (log scale)", color="blue")
        ax1.set_yscale("log")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.grid(True, alpha=0.3)

        # Right y-axis: KDE-based metrics (sample-size independent)
        ax2 = ax1.twinx()
        valid_js = [x for x in kde_js_divergences if not np.isnan(x) and x > 0]
        valid_kl = [x for x in kde_kl_divergences if not np.isnan(x) and x > 0]

        if valid_js:
            js_safe = np.maximum(
                kde_js_divergences, 1e-12
            )  # Ensure positive values for log scale
            ax2.plot(
                calib_sizes_for_metrics[: len(kde_js_divergences)],
                js_safe,
                "r^-",
                linewidth=1.5,
                markersize=4,
                label="JS Divergence",
            )
        if valid_kl and max(valid_kl) < 10:  # Only plot KL if reasonable values
            kl_safe = np.maximum(
                kde_kl_divergences, 1e-12
            )  # Ensure positive values for log scale
            ax2.plot(
                calib_sizes_for_metrics[: len(kde_kl_divergences)],
                kl_safe,
                "g*-",
                linewidth=1.5,
                markersize=4,
                label="KL Divergence",
            )

        ax2.set_ylabel("KDE-based Divergences (log scale)", color="red")
        ax2.set_yscale("log")
        ax2.tick_params(axis="y", labelcolor="red")

        plt.title("Calibration Distribution Convergence\n(EMD vs KDE-based Metrics)")

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        plt.tight_layout()
        plt.show(block=True)
