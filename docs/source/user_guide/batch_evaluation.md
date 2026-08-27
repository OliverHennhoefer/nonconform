---
description: "Generate labeled anomaly batches with oddball and evaluate nonconform discovery procedures without confusing realized metrics with guarantees."
---

# Batch evaluation

The optional [`oddball`](https://github.com/OliverHennhoefer/oddball) package
provides labeled benchmark datasets and reproducible batch generators. It is
useful for exercising a complete evaluation pipeline. Results on its generated
batches are empirical measurements, not proofs of conformal validity or FDR
control.

```bash
pip install "nonconform[data,pyod]"
```

## Generator contract

`BatchGenerator` expects `load_data_func` to return a pandas `DataFrame` with a
`Class` column, where `0` denotes normal and nonzero values denote anomalies.
When using `oddball.load`, pass `as_dataframe=True`.

The generator:

- reserves `train_size` of the normal rows as normal-only training data;
- samples evaluation rows with replacement from the remaining normal pool and
  the anomaly pool;
- yields `(x_batch, y_batch)` pandas objects; and
- uses its own NumPy random generator controlled by `seed`.

!!! warning "`load(...)` returns arrays by default"

    `load(Dataset.WBC)` returns a tuple, which is appropriate for the direct
    setup API but not for `BatchGenerator`. Its loader callback must use
    `load(Dataset.WBC, as_dataframe=True)` so the generator can read `Class`.

## Proportional mode

Proportional mode puts exactly
`int(batch_size * anomaly_proportion)` anomalies in every batch. The integer
conversion rounds down, so the realized proportion can be lower than the
requested value when the product is not integral.

```python
from oddball import BatchGenerator, Dataset, load

generator = BatchGenerator(
    load_data_func=lambda: load(Dataset.WBC, as_dataframe=True),
    batch_size=40,
    anomaly_proportion=0.1,
    anomaly_mode="proportional",
    n_batches=2,
    train_size=0.5,
    seed=42,
)

x_reference = generator.get_training_data()
print("reference shape:", x_reference.shape)

for batch_index, (x_batch, y_batch) in enumerate(generator.generate()):
    print(
        batch_index,
        x_batch.shape,
        int(y_batch.sum()),
    )
```

If `n_batches=None`, proportional generation is unbounded and the caller must
stop iteration. Samples are drawn with replacement, so unbounded generation
does not consume the source pools.

## Probabilistic mode

Despite its name, probabilistic mode targets an exact **global count** over the
configured run. It randomly places
`int(n_batches * batch_size * anomaly_proportion)` anomalies while ensuring the
final total equals that target. Individual batches may have different counts.

```python
from oddball import BatchGenerator, Dataset, load

generator = BatchGenerator(
    load_data_func=lambda: load(Dataset.WBC, as_dataframe=True),
    batch_size=20,
    anomaly_proportion=0.1,
    anomaly_mode="probabilistic",
    n_batches=5,
    seed=42,
)

counts = [int(y_batch.sum()) for _, y_batch in generator.generate()]
print("per-batch anomaly counts:", counts)
print("global anomaly count:", sum(counts))
```

Probabilistic mode requires a finite `n_batches` because the global target must
be known in advance.

## End-to-end discovery evaluation

The next example treats each generated batch as a separate multiple-testing
family. It reports realized FDP and power for each family and then summarizes
those realized values.

```python
import numpy as np
from oddball import BatchGenerator, Dataset, load
from pyod.models.iforest import IForest

from nonconform import ConformalDetector, Split
from nonconform.metrics import false_discovery_rate, statistical_power

generator = BatchGenerator(
    load_data_func=lambda: load(Dataset.SATIMAGE2, as_dataframe=True),
    batch_size=100,
    anomaly_proportion=0.1,
    anomaly_mode="proportional",
    n_batches=5,
    train_size=0.5,
    seed=42,
)

detector = ConformalDetector(
    detector=IForest(n_estimators=50, random_state=42),
    strategy=Split(n_calib=0.3),
    seed=42,
).fit(generator.get_training_data())

rows = []
for batch_index, (x_batch, y_batch) in enumerate(generator.generate()):
    selected = np.asarray(detector.select(x_batch, alpha=0.1))
    y_true = y_batch.to_numpy(dtype=int)
    rows.append(
        {
            "batch": batch_index,
            "discoveries": int(selected.sum()),
            "fdp": float(false_discovery_rate(y_true, selected)),
            "power": float(statistical_power(y_true, selected)),
        }
    )

for row in rows:
    print(row)
print("mean realized FDP:", np.mean([row["fdp"] for row in rows]))
print("mean power:", np.mean([row["power"] for row in rows]))
```

`SATIMAGE2` provides enough normal reference rows for a substantially finer
empirical p-value grid than the small `WBC` demonstration above. This matters:
the smallest classical p-value is `1 / (n_calibration + 1)`, so some small
reference/family combinations cannot satisfy a BH threshold even for an
extreme score.

`false_discovery_rate(...)` computes the realized FDP for the supplied labels
and mask; its historical name is retained for API compatibility. Five generated
families are far too few to establish FDR control, so this example demonstrates
evaluation bookkeeping rather than a theorem check.

## Define the family before generating results

There are two defensible but different designs:

| Design | Procedure | Interpretation |
|---|---|---|
| Each operational batch is its own family | One `select(...)` call per batch | Per-batch FDR target under the relevant assumptions |
| All generated rows form one retrospective family | Concatenate rows, compute p-values, and apply one procedure | FDR target for the combined family |

Do not apply BH independently to chunks of one conceptual family and then
report the pooled discoveries as if one global BH procedure had been used.
Weighted mode is even more batch-sensitive because the target batch is used to
estimate density ratios and WCS couples the complete family.

## Reproducibility

A seed controls the initial training split and sampling sequence. Construct two
generators with identical configuration and seed when you need to demonstrate
full-run reproducibility:

```python
import numpy as np
from oddball import BatchGenerator, Dataset, load

def make_generator() -> BatchGenerator:
    return BatchGenerator(
        load_data_func=lambda: load(Dataset.WBC, as_dataframe=True),
        batch_size=20,
        anomaly_proportion=0.1,
        n_batches=2,
        seed=42,
    )

first = list(make_generator().generate())
second = list(make_generator().generate())

for (x_first, y_first), (x_second, y_second) in zip(
    first,
    second,
    strict=True,
):
    np.testing.assert_array_equal(x_first.to_numpy(), x_second.to_numpy())
    np.testing.assert_array_equal(y_first.to_numpy(), y_second.to_numpy())

print("identical runs")
```

`reset()` restores the generator RNG to the original seed after the training
split has already been constructed. Because construction itself consumed RNG
draws, a reset is not the clearest way to reproduce the first generated run.
Reconstruction, as above, reproduces the entire process.

## Designing a meaningful benchmark

Report enough information to make the evaluation auditable:

- dataset and version;
- normal/anomaly label convention;
- source-pool sizes and whether sampling uses replacement;
- `train_size`, batch size, mode, anomaly proportion, and number of batches;
- detector, conformal strategy, p-value estimator, selection procedure, and
  all seeds;
- per-family discovery count, FDP, and power, including no-discovery families;
- runtime and memory measured on stated hardware;
- uncertainty across independent generator seeds.

Do not use generated test labels to tune a detector and then report the same
batches as final evaluation. Reserve separate generator seeds or source data
for tuning and final reporting, and account for overlap caused by sampling with
replacement.

## Common errors

| Error | Cause | Fix |
|---|---|---|
| `tuple has no attribute columns` | Loader callback used the default array return | Pass `as_dataframe=True` |
| `Expected 'Class' column in data` | Callback returned a DataFrame without the label column | Return the untouched oddball DataFrame or provide the documented schema |
| Not enough normal/anomaly instances | Requested per-batch count exceeds a source pool | Reduce batch count/proportion or choose a larger dataset |
| Probabilistic mode requires `n_batches` | Global target length is undefined | Supply a positive finite batch count |

For ordered evidence and change-detection evaluation, continue with
[Streaming evaluation](streaming_evaluation.md).
