import numpy as np
import pytest

from nonconform._internal import fdp_bounds as core
from nonconform.fdr import FDPCertificate

SUPPORTED_METHODS = ["mc_thc", "mc_hc", "mc_ks", "ks", "mc_bj"]


def _bounds(
    p_values=np.array([0.05, 0.2, 0.2, 0.8]),
    **kwargs,
) -> FDPCertificate:
    params = {
        "n_calibration": 20,
        "confidence": 0.8,
        "n_resamples": 25,
        "seed": 7,
    }
    params.update(kwargs)
    if str(params.get("method", "")).lower() == "ks":
        params.pop("seed", None)
        params.pop("n_resamples", None)
    return FDPCertificate.from_p_values(p_values, **params)


@pytest.mark.parametrize(
    ("kwargs", "error_type", "match"),
    [
        ({"p_values": np.array([[0.1, 0.2]])}, ValueError, "1D"),
        ({"p_values": np.array([0.1, np.nan])}, ValueError, "finite"),
        ({"p_values": np.array([0.1, 1.2])}, ValueError, "within"),
        ({"p_values": np.array([])}, ValueError, "at least one"),
        ({"n_calibration": 0}, ValueError, "positive"),
        ({"confidence": 1.0}, ValueError, "confidence"),
        ({"n_resamples": 0}, ValueError, "positive"),
        ({"lower": 0.2, "upper": 0.1}, ValueError, "lower"),
        ({"beta": 0.0}, ValueError, "beta"),
        ({"precision": 0.0, "method": "mc_bj"}, ValueError, "precision"),
        ({"method": "marginal_mc"}, ValueError, "method"),
    ],
)
def test_conformal_fdp_upper_bound_validates_inputs(kwargs, error_type, match):
    params = {
        "p_values": np.array([0.1, 0.2, 0.6]),
        "n_calibration": 10,
        "confidence": 0.8,
        "n_resamples": 5,
        "seed": 1,
    }
    params.update(kwargs)
    with pytest.raises(error_type, match=match):
        FDPCertificate.from_p_values(**params)


def test_conformal_fdp_upper_bound_rejects_non_string_method():
    with pytest.raises(TypeError, match="method"):
        _bounds(method=1)


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("MC-THC", "mc_thc"),
        ("MC HC", "mc_hc"),
        ("mc-ks", "mc_ks"),
        ("KS", "ks"),
        ("MC-BJ", "mc_bj"),
    ],
)
def test_method_aliases_normalize(alias, expected):
    result = _bounds(method=alias)

    assert result.method == expected


def test_default_thresholds_are_sorted_unique_p_values():
    result = _bounds()

    np.testing.assert_allclose(result.thresholds, np.array([0.05, 0.2, 0.8]))
    np.testing.assert_array_equal(result.rejection_counts, np.array([1, 3, 4]))


def test_threshold_queries_preserve_order_and_zero_discoveries():
    thresholds = np.array([0.0, 0.5, 0.1])
    result = _bounds()
    table = result.to_frame(thresholds)
    np.testing.assert_allclose(table.threshold, thresholds)
    np.testing.assert_array_equal(table.discoveries, [0, 3, 1])
    assert table.fdp_upper_bound[0] == 0.0


def test_certificate_isolates_source_and_returned_state():
    p_values = np.array([0.4, 0.1, 0.8])
    result = _bounds(p_values=p_values)
    expected = result.to_frame()
    p_values[:] = 0.99
    for name in [
        "p_values",
        "thresholds",
        "rejection_counts",
        "fdp_upper_bounds",
        "precision_lower_bounds",
    ]:
        arr = getattr(result, name)
        with pytest.raises(ValueError):
            arr[:] = 0
        with pytest.raises(ValueError):
            arr.flags.writeable = True
        with pytest.raises(AttributeError):
            setattr(result, name, np.array([0]))
    for name in ["method", "confidence", "boost", "seed", "lower"]:
        with pytest.raises(AttributeError):
            setattr(result, name, None)
    # Read-only NumPy bytes still allow shape/dtype metadata edits on a returned
    # array. Those edits must never touch the certificate's owned array object.
    for name in ["p_values", "thresholds", "rejection_counts"]:
        exposed = getattr(result, name)
        exposed.shape = (1, exposed.size)
        exposed.dtype = np.uint8
        assert getattr(result, name).ndim == 1
        assert getattr(result, name).dtype != np.uint8
    frame = result.to_frame()
    frame.iloc[:, :] = 0
    np.testing.assert_array_equal(result.to_frame(), expected)
    result.bound_at([0.1, 0.8])[:] = 0
    result.select(0.2)[:] = False
    np.testing.assert_array_equal(result.to_frame(), expected)
    np.testing.assert_array_equal(result.p_values, [0.4, 0.1, 0.8])


def test_bound_at_supports_scalar_and_vector_thresholds():
    result = _bounds()

    scalar_bound = result.bound_at(0.5)
    vector_bounds = result.bound_at(np.array([0.1, 0.5]))

    assert isinstance(scalar_bound, float)
    np.testing.assert_allclose(
        vector_bounds, result.to_frame([0.1, 0.5]).fdp_upper_bound
    )


def test_precision_lower_bounds_are_fdp_complements():
    result = _bounds()

    np.testing.assert_allclose(
        result.precision_lower_bounds,
        1.0 - result.fdp_upper_bounds,
    )
    np.testing.assert_allclose(
        result.precision_at(result.thresholds),
        result.precision_lower_bounds,
    )
    assert isinstance(result.precision_at(0.5), float)


def test_to_frame_returns_threshold_level_certificate_table():
    result = _bounds()

    table = result.to_frame()

    assert list(table.columns) == [
        "threshold",
        "discoveries",
        "fdp_upper_bound",
        "precision_lower_bound",
    ]
    np.testing.assert_allclose(table["threshold"].to_numpy(), result.thresholds)
    np.testing.assert_array_equal(
        table["discoveries"].to_numpy(),
        result.rejection_counts,
    )
    np.testing.assert_allclose(
        table["fdp_upper_bound"].to_numpy(),
        result.fdp_upper_bounds,
    )
    np.testing.assert_allclose(
        table["precision_lower_bound"].to_numpy(),
        result.precision_lower_bounds,
    )


def test_to_frame_accepts_custom_threshold_grid():
    result = _bounds()
    grid = np.array([0.0, 0.2, 0.6])

    table = result.to_frame(thresholds=grid)

    np.testing.assert_allclose(table["threshold"].to_numpy(), grid)
    np.testing.assert_array_equal(table["discoveries"].to_numpy(), np.array([0, 3, 3]))
    np.testing.assert_allclose(
        table["fdp_upper_bound"].to_numpy(),
        result.bound_at(grid),
    )
    np.testing.assert_allclose(
        table["precision_lower_bound"].to_numpy(),
        result.precision_at(grid),
    )


def test_select_returns_original_order_threshold_mask():
    p_values = np.array([0.4, 0.1, 0.8, 0.2])
    result = _bounds(p_values=p_values)

    np.testing.assert_array_equal(
        result.select(0.2),
        np.array([False, True, False, True]),
    )


def test_select_rejects_vector_thresholds():
    result = _bounds()

    with pytest.raises(ValueError, match="scalar"):
        result.select(np.array([0.1, 0.2]))


def test_same_seed_gives_identical_bounds():
    first = _bounds(seed=11, method="mc_bj")
    second = _bounds(seed=11, method="mc_bj")

    np.testing.assert_allclose(first.fdp_upper_bounds, second.fdp_upper_bounds)


def test_different_seed_can_change_bounds():
    first = _bounds(
        p_values=np.array([0.01, 0.03, 0.08, 0.25, 0.6]),
        confidence=0.5,
        n_resamples=50,
        seed=1,
    )
    alternatives = [
        _bounds(
            p_values=np.array([0.01, 0.03, 0.08, 0.25, 0.6]),
            confidence=0.5,
            n_resamples=50,
            seed=seed,
        )
        for seed in range(2, 8)
    ]

    assert any(
        not np.allclose(first.fdp_upper_bounds, other.fdp_upper_bounds)
        for other in alternatives
    )


@pytest.mark.parametrize("method", SUPPORTED_METHODS)
def test_supported_methods_return_valid_bounds(method):
    result = _bounds(
        p_values=np.array([0.001, 0.01, 0.03, 0.08, 0.25, 0.6]),
        n_calibration=50,
        confidence=0.5,
        n_resamples=20,
        seed=3,
        method=method,
    )

    assert result.method == method
    assert np.all((0.0 <= result.fdp_upper_bounds) & (result.fdp_upper_bounds <= 1.0))
    np.testing.assert_allclose(
        result.precision_lower_bounds,
        1.0 - result.fdp_upper_bounds,
    )


def test_supported_methods_match_fixed_reference_values():
    p_values = np.array([0.001, 0.01, 0.03, 0.08, 0.25, 0.6])
    thresholds = np.array([0.001, 0.01, 0.03, 0.08, 0.25, 0.6])
    expected = {
        "mc_thc": np.array(
            [
                0.406804935815,
                0.203402467907,
                0.258195099608,
                0.356399518904,
                0.485119615123,
                0.570933012602,
            ]
        ),
        "mc_hc": np.array(
            [
                0.116166719315,
                0.203402467907,
                0.258195099608,
                0.356399518904,
                0.485119615123,
                0.570933012602,
            ]
        ),
        "mc_ks": np.array(
            [
                1.0,
                0.748581325403,
                0.539054216936,
                0.479290662702,
                0.583432530161,
                0.652860441801,
            ]
        ),
        "ks": np.array(
            [
                1.0,
                1.0,
                0.821888027249,
                0.691416020437,
                0.753132816349,
                0.794277346958,
            ]
        ),
        "mc_bj": np.array(
            [
                0.0,
                0.0,
                0.333333333333,
                0.25,
                0.4,
                0.5,
            ]
        ),
    }

    for method, expected_bounds in expected.items():
        result = _bounds(
            p_values,
            n_calibration=50,
            confidence=0.5,
            n_resamples=20,
            seed=3,
            method=method,
        )

        np.testing.assert_allclose(
            result.bound_at(thresholds), expected_bounds, rtol=1e-10
        )


def test_global_numpy_rng_state_is_not_mutated():
    np.random.seed(123)  # noqa: NPY002
    _bounds()
    observed = np.random.random(4)  # noqa: NPY002

    np.random.seed(123)  # noqa: NPY002
    expected = np.random.random(4)  # noqa: NPY002
    np.testing.assert_allclose(observed, expected)


@pytest.mark.parametrize("method", SUPPORTED_METHODS)
def test_boosted_bounds_are_no_larger_than_unboosted_bounds(method):
    p_values = np.array([0.01, 0.03, 0.06, 0.4, 0.8])

    boosted = _bounds(
        p_values=p_values,
        seed=9,
        boost=True,
        method=method,
    )
    unboosted = _bounds(
        p_values=p_values,
        seed=9,
        boost=False,
        method=method,
    )

    assert np.all(boosted.fdp_upper_bounds <= unboosted.fdp_upper_bounds + 1e-12)


@pytest.mark.parametrize("method", SUPPORTED_METHODS)
@pytest.mark.parametrize("boost", [False, True])
@pytest.mark.parametrize("p_values", [[0.0], [1.0], [0.4, 0.1], [0, 0.1, 0.1, 0.7, 1]])
def test_prepared_queries_match_original_loop(method, boost, p_values):
    """Independent old evaluator, including inclusive ties and arbitrary grids."""
    certificate = _bounds(np.array(p_values), method=method, boost=boost)
    grid = np.array([1, 0.2, 0, 0.1, 0.7, 0.1, 0.99])
    sorted_p = np.sort(p_values)
    m = len(p_values)
    numerator = np.full(len(grid), m, dtype=float)
    max_p = np.zeros(len(grid))
    if boost:
        for p in sorted_p:
            mask = p <= grid
            max_p[mask] = np.maximum(max_p[mask], p)
            second = m * certificate._envelope.evaluate(np.array([p]))[0]
            second -= np.count_nonzero(sorted_p <= p)
            numerator[mask] = np.minimum(numerator[mask], second)
        numerator += np.searchsorted(sorted_p, max_p, side="right")
    else:
        numerator = m * certificate._envelope.evaluate(grid)
    counts = np.searchsorted(sorted_p, grid, side="right")
    expected = np.clip(
        np.divide(numerator, counts, out=np.zeros(len(grid)), where=counts > 0), 0, 1
    )
    np.testing.assert_array_equal(certificate.bound_at(grid), expected)
    for threshold, bound in zip(grid, expected, strict=True):
        assert certificate.bound_at(threshold) == bound
    assert certificate.bound_at([]).shape == (0,)
    assert certificate.to_frame([]).shape == (0, 4)


@pytest.mark.parametrize("boost", [False, True])
def test_infinite_hc_cutoff_is_conservative_at_endpoints(boost):
    with np.errstate(invalid="raise"):
        certificate = _bounds(
            np.array([0.1, 0.2, 1]),
            n_calibration=100,
            confidence=0.95,
            n_resamples=10,
            seed=1,
            method="mc_hc",
            boost=boost,
        )
        assert np.isposinf(certificate._envelope.summary_quantile)
        np.testing.assert_array_equal(certificate.bound_at([0, 0.2, 1]), [0, 1, 1])
        with_zero = _bounds(
            np.array([0, 1]),
            confidence=0.95,
            n_resamples=10,
            method="mc_hc",
            boost=boost,
        )
        np.testing.assert_array_equal(with_zero.bound_at([0, 1]), [1, 1])


@pytest.mark.parametrize("boost", [False, True])
@pytest.mark.parametrize("method", SUPPORTED_METHODS)
def test_queries_never_sort_or_resample(monkeypatch, boost, method):
    certificate = _bounds(method=method, boost=boost)

    def forbidden(*args, **kwargs):
        pytest.fail("query sorted or resampled after certificate construction")

    monkeypatch.setattr(np, "sort", forbidden)
    monkeypatch.setattr(np, "unique", forbidden)
    monkeypatch.setattr(core, "_sample_conformal_null_p_values", forbidden)
    for _ in range(2):
        certificate.bound_at(0.1)
        certificate.precision_at([0.5, 0.1])
        certificate.to_frame()
        certificate.to_frame([0, 0.1, 1])
        certificate.select(0.2)
        _ = certificate.fdp_upper_bounds, certificate.precision_lower_bounds


@pytest.mark.parametrize(
    "method,option",
    [
        ("ks", {"seed": 1}),
        ("ks", {"n_resamples": 5}),
        ("mc_hc", {"lower": 0.1}),
        ("mc_ks", {"upper": 0.5}),
        ("mc_bj", {"beta": 0.5}),
        ("mc_thc", {"precision": 1e-8}),
    ],
)
def test_reject_inapplicable_options(method, option):
    with pytest.raises(ValueError, match="does not apply"):
        FDPCertificate.from_p_values([0.1], n_calibration=10, method=method, **option)


def test_effective_options_and_removed_api():
    import nonconform.fdr as fdr

    for name in [
        "FDPBoundResult",
        "conformal_fdp_upper_bound",
        "conformal_fdp_upper_bound_from_result",
    ]:
        assert not hasattr(fdr, name)
        assert name not in fdr.__all__
    certificate = FDPCertificate.from_p_values([0.1], n_calibration=10, seed=1)
    assert certificate.method == "mc_thc"
    assert (certificate.n_resamples, certificate.confidence, certificate.boost) == (
        1000,
        0.95,
        True,
    )
    assert (certificate.lower, certificate.upper, certificate.beta) == (0.01, 0.99, 0.5)
    assert certificate.precision is None
    ks = FDPCertificate.from_p_values([0.1], n_calibration=10, method="ks")
    assert (ks.n_resamples, ks.seed, ks.lower, ks.upper, ks.beta, ks.precision) == (
        None,
    ) * 6
    bj = _bounds(method="mc_bj")
    assert bj.precision == 1e-8
    with pytest.raises(TypeError, match="from_p_values"):
        FDPCertificate()
    with pytest.raises(TypeError, match="thresholds"):
        _bounds(thresholds=[0.1])


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan, [[0.1]], "invalid"])
def test_query_validation(threshold):
    certificate = _bounds()
    for query in [certificate.bound_at, certificate.precision_at, certificate.select]:
        with pytest.raises(ValueError):
            query(threshold)
