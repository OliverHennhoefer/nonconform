"""PyOD's default threshold versus nonconform FDR control on seven datasets.

Reproduces the comparison table in the README. For each dataset and 20 random
splits, Isolation Forests with the same settings are thresholded two ways
(the PyOD baseline is fit on all training rows; ``Split`` withholds ``n_calib``):

- PyOD ``predict()``, which flags the top 10% of scores (``contamination=0.1``).
- ``ConformalDetector.select(alpha=0.1)``, which targets a 10% false discovery rate.

Requires ``pip install "nonconform[pyod,data]"``. Runtime is a few minutes.
"""

import logging
import warnings

import numpy as np
from oddball import Dataset, load
from pyod.models.iforest import IForest

from nonconform import ConformalDetector, Split
from nonconform.metrics import false_discovery_rate, statistical_power

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
SEEDS = range(20)


def evaluate(y, flagged):
    flagged = np.asarray(flagged, dtype=bool)
    return (
        flagged.sum(),
        false_discovery_rate(y=y, y_hat=flagged),
        statistical_power(y=y, y_hat=flagged),
    )


print(
    f"{'dataset':<12} {'anomalies':>9} | {'PyOD flagged':>12} {'false':>6} "
    f"{'recall':>6} | {'nonconform flagged':>18} {'false':>6} {'recall':>6}"
)
for dataset in DATASETS:
    pyod, conformal = [], []
    for seed in SEEDS:
        x_train, x_test, y_test = load(dataset, setup=True, seed=seed)
        y = np.asarray(y_test, dtype=bool)

        baseline = IForest(random_state=seed).fit(x_train)
        pyod.append(evaluate(y, baseline.predict(x_test)))

        # Size calibration so one anomaly can reach BH's first cutoff, alpha / m.
        n_calib = min(int(len(x_test) / ALPHA), len(x_train) // 2)
        detector = ConformalDetector(
            detector=IForest(random_state=seed),
            strategy=Split(n_calib=n_calib),
            seed=seed,
        ).fit(x_train)
        conformal.append(evaluate(y, detector.select(x_test, alpha=ALPHA)))

    p, c = np.mean(pyod, axis=0), np.mean(conformal, axis=0)
    print(
        f"{dataset.name:<12} {int(y.sum()):>9} | {p[0]:>12.0f} {p[1]:>6.0%} "
        f"{p[2]:>6.0%} | {c[0]:>18.0f} {c[1]:>6.0%} {c[2]:>6.0%}"
    )
