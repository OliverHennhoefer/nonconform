"""Three ways to turn Isolation Forest scores into flags, on seven datasets.

Reproduces the table and figure on the "Default thresholds" benchmark page of
the documentation. For each dataset and 20 random splits, Isolation Forests
with the same settings are thresholded three ways:

- PyOD ``predict()``, which flags scores above the 90th percentile of the
  training scores (``contamination=0.1``). The model is fit on all training rows.
- ``ConformalDetector.select(alpha=0.1)``, which targets a 10% false discovery
  rate: on average, at most 10% of the flagged points are normal.
- ``ConformalDetector.fpr_bounds(confidence=0.9)`` with ``target_fpr=0.1``,
  a score threshold that, with 90% confidence, flags at most 10% of normal points.

Both nonconform rules use the same fitted detector, which withholds ``n_calib``
training rows for calibration.

Requires ``pip install "nonconform[pyod,data]" matplotlib``. Runtime is a few
minutes. Pass ``--plot PATH`` to write the figure.
"""

import argparse
import logging
import warnings

import numpy as np
from oddball import Dataset, load
from pyod.models.iforest import IForest

from nonconform import ConformalDetector, Split

warnings.filterwarnings("ignore")
logging.getLogger("nonconform").setLevel(logging.ERROR)

DATASETS = [
    Dataset.SHUTTLE,
    Dataset.FRAUD,
    Dataset.MAMMOGRAPHY,
    Dataset.SATIMAGE2,
    Dataset.THYROID,
    Dataset.CARDIO,
    Dataset.HTTP,
]
ALPHA = 0.1
TARGET_FPR = 0.1
CONFIDENCE = 0.9
SEEDS = range(20)
RULES = ["contamination=0.1", "select(alpha=0.1)", "fpr_bounds(target 10%)"]
NAMES = {
    "SHUTTLE": "Shuttle",
    "FRAUD": "Fraud",
    "MAMMOGRAPHY": "Mammography",
    "SATIMAGE2": "Satimage-2",
    "THYROID": "Thyroid",
    "CARDIO": "Cardio",
    "HTTP": "HTTP",
}


def evaluate(y, flagged):
    """Return flagged count, false share, recall and normal-flag rate."""
    flagged = np.asarray(flagged, dtype=bool)
    true_hits = int((flagged & y).sum())
    false_hits = int((flagged & ~y).sum())
    return (
        flagged.sum(),
        false_hits / max(flagged.sum(), 1),
        true_hits / y.sum(),
        false_hits / (~y).sum(),
        true_hits,
        false_hits,
    )


def run():
    results = {}
    for dataset in DATASETS:
        runs = {rule: [] for rule in RULES}
        for seed in SEEDS:
            x_train, x_test, y_test = load(dataset, setup=True, seed=seed)
            y = np.asarray(y_test, dtype=bool)

            baseline = IForest(random_state=seed).fit(x_train)
            runs[RULES[0]].append(evaluate(y, baseline.predict(x_test)))

            # Size calibration so one anomaly can reach BH's first cutoff, alpha / m.
            n_calib = min(int(len(x_test) / ALPHA), len(x_train) // 2)
            detector = ConformalDetector(
                detector=IForest(random_state=seed),
                strategy=Split(n_calib=n_calib),
                seed=seed,
            ).fit(x_train)
            runs[RULES[1]].append(evaluate(y, detector.select(x_test, alpha=ALPHA)))

            certificate = detector.fpr_bounds(confidence=CONFIDENCE)
            flagged = certificate.select(
                detector.score_samples(x_test), target_fpr=TARGET_FPR
            )
            runs[RULES[2]].append(evaluate(y, flagged))
        results[dataset.name] = {
            "anomalies": int(y.sum()),
            "test": len(y),
            **{rule: np.mean(values, axis=0) for rule, values in runs.items()},
            # Splits whose normal-flag rate on the test set exceeds the target.
            "over_target": {
                rule: int(sum(v[3] > TARGET_FPR for v in values))
                for rule, values in runs.items()
            },
        }
    return results


def print_table(results):
    print(
        f"{'dataset':<12} {'test':>5} {'anom':>5} | {'rule':<23} "
        f"{'flagged':>7} {'false':>6} {'recall':>6} {'normals':>8} {'>10%':>5}"
    )
    for name, row in results.items():
        for rule in RULES:
            flagged, false_share, recall, fpr = row[rule][:4]
            print(
                f"{name:<12} {row['test']:>5} {row['anomalies']:>5} | {rule:<23} "
                f"{flagged:>7.0f} {false_share:>6.0%} {recall:>6.0%} {fpr:>8.1%} "
                f"{row['over_target'][rule]:>5}"
            )


def plot(results, path):
    import matplotlib.pyplot as plt

    found, false, ink, muted = "#eb6834", "#b9b8b2", "#0b0b0b", "#8a8984"
    plt.rcParams.update(
        {
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "axes.edgecolor": muted,
            "axes.labelcolor": ink,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": ink,
            "ytick.color": ink,
            "legend.frameon": False,
            "font.size": 9,
        }
    )
    names = list(results)
    fig, axes = plt.subplots(len(names), 1, figsize=(8, 9), sharex=False)
    labels = ["contamination=0.1", "select, FDR 10%", "fpr_bounds, FPR 10%"]
    for ax, name in zip(axes, names, strict=True):
        row = results[name]
        hits = [row[rule][4] for rule in RULES]
        misses = [row[rule][5] for rule in RULES]
        ypos = np.arange(len(RULES))[::-1]
        ax.barh(ypos, hits, color=found, height=0.7, label="true anomalies flagged")
        ax.barh(ypos, misses, left=hits, color=false, height=0.7, label="false alarms")
        ax.axvline(row["anomalies"], color=ink, lw=1, ls="--")
        ax.set_yticks(ypos, labels)
        ax.set_title(
            f"{NAMES[name]}: {row['anomalies']} anomalies among {row['test']} test points",
            loc="left",
            fontsize=9,
            color=ink,
        )
        ax.set_xlim(0, max(max(h + m for h, m in zip(hits, misses)), 1) * 1.1)
        ax.tick_params(axis="y", length=0)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper right", ncols=2, fontsize=8)
    axes[-1].set_xlabel(
        "points flagged (mean of 20 splits); dashed line: anomalies present"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot", help="write the figure to this path")
    args = parser.parse_args()
    results = run()
    print_table(results)
    if args.plot:
        plot(results, args.plot)
