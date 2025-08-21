import gzip
import io
import os
import shutil
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from nonconform.utils.func.logger import get_logger

logger = get_logger("utils.data.load")

DATASET_VERSION = os.environ.get("UNQUAD_DATASET_VERSION", "v.0.8.1-datasets")
DATASET_BASE_URL = os.environ.get(
    "UNQUAD_DATASET_URL",
    f"https://github.com/OliverHennhoefer/nonconform/releases/download/{DATASET_VERSION}/",
)
DATASET_SUFFIX = ".parquet.gz"
_DATASET_CACHE: dict[str, bytes] = {}  # In-memory cache for downloaded datasets

# Disk cache directory (version-aware) - created lazily
_CACHE_DIR = None


def _get_cache_dir() -> Path:
    """Get cache directory, creating it lazily."""
    global _CACHE_DIR
    if _CACHE_DIR is None:
        _CACHE_DIR = (
            Path(
                os.environ.get(
                    "UNQUAD_CACHE_DIR", Path.home() / ".cache" / "nonconform"
                )
            )
            / DATASET_VERSION
        )
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


# Check if pyarrow is available for reading parquet files
try:
    import pyarrow  # noqa: F401

    _PYARROW_AVAILABLE = True
except ImportError:
    _PYARROW_AVAILABLE = False


def load_breast(
    setup: bool = False, seed: int | None = None
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the Breast Cancer Wisconsin (Diagnostic) dataset.

    This dataset contains features computed from a digitized image of a
    fine needle aspirate (FNA) of a breast mass. They describe
    characteristics of the cell nuclei present in the image. The "Class"
    column typically indicates malignant (1) or benign (0).

    Args:
        setup (bool, optional): If ``True``, splits the data into training
            and testing sets according to a specific experimental setup,
            returning (x_train, x_test, y_test). `x_train` contains only
            normal samples. Defaults to ``False``, which returns the full
            DataFrame.
        seed (int, optional): Seed for random number generation used
            in data splitting if `setup` is ``True``. Defaults to ``None``
            for true randomness.

    Returns
    -------
        Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
            If `setup` is ``False``, returns the complete dataset as a
            DataFrame. If `setup` is ``True``, returns a tuple:
            (x_train, x_test, y_test).
    """
    return _load_dataset(Path(f"breast{DATASET_SUFFIX}"), setup, seed)


def load_fraud(
    setup: bool = False, seed: int | None = None
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load a credit card fraud detection dataset.

    This dataset typically contains transactions made by European cardholders.
    It presents transactions that occurred in two days, where it has features
    that are numerical input variables, the result of a PCA transformation.
    The "Class" column indicates fraudulent (1) or legitimate (0) transactions.

    Args:
        setup (bool, optional): If ``True``, splits the data into training
            and testing sets according to a specific experimental setup,
            returning (x_train, x_test, y_test). `x_train` contains only
            normal samples. Defaults to ``False``, which returns the full
            DataFrame.
        seed (int, optional): Seed for random number generation used
            in data splitting if `setup` is ``True``. Defaults to ``None``
            for true randomness.

    Returns
    -------
        Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
            If `setup` is ``False``, returns the complete dataset as a
            DataFrame. If `setup` is ``True``, returns a tuple:
            (x_train, x_test, y_test).
    """
    return _load_dataset(Path(f"fraud{DATASET_SUFFIX}"), setup, seed)


def load_ionosphere(
    setup: bool = False, seed: int | None = None
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the Ionosphere dataset.

    This radar data was collected by a system in Goose Bay, Labrador.
    The targets were free electrons in the ionosphere. "Good" radar returns
    are those showing evidence of some type of structure in the ionosphere.
    "Bad" returns are those that do not; their signals pass through the
    ionosphere. The "Class" column indicates "good" (0) or "bad" (1, anomaly).

    Args:
        setup (bool, optional): If ``True``, splits the data into training
            and testing sets according to a specific experimental setup,
            returning (x_train, x_test, y_test). `x_train` contains only
            normal samples. Defaults to ``False``, which returns the full
            DataFrame.
        seed (int, optional): Seed for random number generation used
            in data splitting if `setup` is ``True``. Defaults to ``None``
            for true randomness.

    Returns
    -------
        Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
            If `setup` is ``False``, returns the complete dataset as a
            DataFrame. If `setup` is ``True``, returns a tuple:
            (x_train, x_test, y_test).
    """
    return _load_dataset(Path(f"ionosphere{DATASET_SUFFIX}"), setup, seed)


def load_mammography(
    setup: bool = False, seed: int | None = None
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the Mammography dataset.

    This dataset is used for detecting breast cancer based on mammographic
    findings. It contains features related to BI-RADS assessment, age,
    shape, margin, and density. The "Class" column usually indicates
    benign (0) or malignant (1, anomaly).

    Args:
        setup (bool, optional): If ``True``, splits the data into training
            and testing sets according to a specific experimental setup,
            returning (x_train, x_test, y_test). `x_train` contains only
            normal samples. Defaults to ``False``, which returns the full
            DataFrame.
        seed (int, optional): Seed for random number generation used
            in data splitting if `setup` is ``True``. Defaults to ``None``
            for true randomness.

    Returns
    -------
        Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
            If `setup` is ``False``, returns the complete dataset as a
            DataFrame. If `setup` is ``True``, returns a tuple:
            (x_train, x_test, y_test).
    """
    return _load_dataset(Path(f"mammography{DATASET_SUFFIX}"), setup, seed)


def load_musk(
    setup: bool = False, seed: int | None = None
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the Musk (Version 2) dataset.

    This dataset describes a set of 102 molecules of which 39 are judged
    by human experts to be musks and the remaining 63 molecules are
    judged to be non-musks. The 166 features describe the three-dimensional
    conformation of the molecules. The "Class" indicates musk (1, anomaly)
    or non-musk (0).

    Args:
        setup (bool, optional): If ``True``, splits the data into training
            and testing sets according to a specific experimental setup,
            returning (x_train, x_test, y_test). `x_train` contains only
            normal samples. Defaults to ``False``, which returns the full
            DataFrame.
        seed (int, optional): Seed for random number generation used
            in data splitting if `setup` is ``True``. Defaults to ``None``
            for true randomness.

    Returns
    -------
        Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
            If `setup` is ``False``, returns the complete dataset as a
            DataFrame. If `setup` is ``True``, returns a tuple:
            (x_train, x_test, y_test).
    """
    return _load_dataset(Path(f"musk{DATASET_SUFFIX}"), setup, seed)


def load_shuttle(
    setup: bool = False, seed: int | None = None
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the Shuttle dataset.

    This dataset contains data from a NASA space shuttle mission concerning
    the position of radiators in the shuttle. The "Class" column indicates
    normal (0) or anomalous (1) states. The original dataset has multiple
    anomaly classes; this version typically simplifies it to binary.

    Args:
        setup (bool, optional): If ``True``, splits the data into training
            and testing sets according to a specific experimental setup,
            returning (x_train, x_test, y_test). `x_train` contains only
            normal samples. Defaults to ``False``, which returns the full
            DataFrame.
        seed (int, optional): Seed for random number generation used
            in data splitting if `setup` is ``True``. Defaults to ``None``
            for true randomness.

    Returns
    -------
        Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
            If `setup` is ``False``, returns the complete dataset as a
            DataFrame. If `setup` is ``True``, returns a tuple:
            (x_train, x_test, y_test).
    """
    return _load_dataset(Path(f"shuttle{DATASET_SUFFIX}"), setup, seed)


def load_thyroid(
    setup: bool = False, seed: int | None = None
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the Thyroid Disease (ann-thyroid) dataset.

    This dataset is used for diagnosing thyroid conditions based on patient
    attributes and test results. The "Class" column indicates normal (0)
    or some form of thyroid disease (1, anomaly).

    Args:
        setup (bool, optional): If ``True``, splits the data into training
            and testing sets according to a specific experimental setup,
            returning (x_train, x_test, y_test). `x_train` contains only
            normal samples. Defaults to ``False``, which returns the full
            DataFrame.
        seed (int, optional): Seed for random number generation used
            in data splitting if `setup` is ``True``. Defaults to ``None``
            for true randomness.

    Returns
    -------
        Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
            If `setup` is ``False``, returns the complete dataset as a
            DataFrame. If `setup` is ``True``, returns a tuple:
            (x_train, x_test, y_test).
    """
    return _load_dataset(Path(f"thyroid{DATASET_SUFFIX}"), setup, seed)


def load_wbc(
    setup: bool = False, seed: int | None = None
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the Wisconsin Breast Cancer (Original) (WBC) dataset.

    This dataset contains features derived from clinical observations of
    breast cancer, such as clump thickness, cell size uniformity, etc.
    The "Class" column indicates benign (0) or malignant (1, anomaly).
    Note: This is distinct from the "breast" dataset which is often the
    Diagnostic version.

    Args:
        setup (bool, optional): If ``True``, splits the data into training
            and testing sets according to a specific experimental setup,
            returning (x_train, x_test, y_test). `x_train` contains only
            normal samples. Defaults to ``False``, which returns the full
            DataFrame.
        seed (int, optional): Seed for random number generation used
            in data splitting if `setup` is ``True``. Defaults to ``None``
            for true randomness.

    Returns
    -------
        Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
            If `setup` is ``False``, returns the complete dataset as a
            DataFrame. If `setup` is ``True``, returns a tuple:
            (x_train, x_test, y_test).
    """
    return _load_dataset(Path(f"wbc{DATASET_SUFFIX}"), setup, seed)


def _download_dataset(filename: str, show_progress: bool = True) -> io.BytesIO:
    """Download dataset with memory and disk caching.

    Args:
        filename: Name of the dataset file (e.g., "breast.parquet.gz")
        show_progress: Whether to show download progress

    Returns
    -------
        BytesIO object containing the compressed dataset

    Raises
    ------
        URLError: If download fails
    """
    # Check memory cache first
    if filename in _DATASET_CACHE:
        return io.BytesIO(_DATASET_CACHE[filename])

    # Check disk cache second
    cache_dir = _get_cache_dir()
    cache_file = cache_dir / filename
    if cache_file.exists():
        logger.debug("Loading %s from disk cache (v%s)", filename, DATASET_VERSION)
        with open(cache_file, "rb") as f:
            data = f.read()
        _DATASET_CACHE[filename] = data
        return io.BytesIO(data)

    # Clean old versions before downloading
    _cleanup_old_versions()

    # Download file
    url = urljoin(DATASET_BASE_URL, filename)
    logger.info("Downloading %s from %s...", filename, url)

    try:
        # Add headers to avoid GitHub rate limiting
        req = Request(url, headers={"User-Agent": "nonconform-dataset-loader"})

        with urlopen(req) as response:
            total_size = int(response.headers.get("Content-Length", 0))

            # Download with progress bar if requested
            if show_progress and total_size > 0:
                try:
                    with tqdm(
                        total=total_size, unit="B", unit_scale=True, desc=filename
                    ) as pbar:
                        # Use bytearray for efficient concatenation, then convert
                        data_buffer = bytearray()
                        while True:
                            chunk = response.read(8192)
                            if not chunk:
                                break
                            data_buffer.extend(chunk)
                            pbar.update(len(chunk))
                        data = bytes(data_buffer)
                except ImportError:
                    # Fallback to simple download without progress bar
                    data = response.read()
            else:
                data = response.read()

    except URLError as e:
        raise URLError(f"Failed to download {filename}: {e!s}") from e

    # Cache in memory and on disk
    _DATASET_CACHE[filename] = data
    # Ensure cache directory exists before writing (important for tests)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        f.write(data)

    logger.debug("Successfully cached %s (%.1f KB)", filename, len(data) / 1024)
    return io.BytesIO(data)


def _load_dataset(
    file_path, setup: bool, seed: int | None
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load a dataset from a gzipped Parquet file and optionally sets it up.

    This is a helper function used by the specific dataset loaders. It downloads
    the dataset from GitHub releases and caches it in memory, then reads the
    Parquet file compressed with gzip. If `setup` is true, it calls
    `_create_setup` to split the data.

    Args:
        file_path: The path object (used to extract filename).
        setup (bool): If ``True``, the data is processed by `_create_setup`
            to produce training and testing sets.
        seed (int | None): Seed for random number generation, used if
            `setup` is ``True``. None for true randomness.

    Returns
    -------
        Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
            If `setup` is ``False``, returns the complete dataset as a
            DataFrame. If `setup` is ``True``, returns a tuple:
            (x_train, x_test, y_test).

    Raises
    ------
        ImportError: If pyarrow is not available for reading parquet files.
        URLError: If dataset download fails.
    """
    if not _PYARROW_AVAILABLE:
        raise ImportError(
            "The datasets functionality requires pyarrow to read parquet files. "
            "Please install the data dependencies with: pip install nonconform[data]"
        )

    # Extract filename from the provided path
    filename = file_path.name

    # Download dataset to memory
    compressed_stream = _download_dataset(filename)

    # GzipFile is not seekable, which is required by pyarrow.
    # Decompress to an in-memory buffer to allow seeking.
    with gzip.open(compressed_stream, "rb") as f_gz:
        buffer = io.BytesIO(f_gz.read())
    df = pd.read_parquet(buffer)

    if setup:
        return _create_setup(df, seed=seed)

    return df


def _create_setup(
    df: pd.DataFrame, seed: int | None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Create an experimental train/test split from a dataset.

    This setup aims to create a scenario for anomaly detection where:
    - The training set (`x_train`) contains only normal samples (Class 0).
    - The test set (`x_test`, `y_test`) contains a mix of normal samples
      and a smaller proportion of outlier samples (Class 1).

    The sizes are determined as follows:
    - `n_train`: Half of the available normal samples.
    - `n_test`: The minimum of 1000 or one-third of `n_train`.
    - `n_test_outlier`: 10% of `n_test`.
    - `n_test_normal`: The remaining 90% of `n_test`.

    The "Class" column is dropped from `x_train` and `x_test`.

    Args:
        df (pandas.DataFrame): The input DataFrame, expected to have a "Class"
            column where 0 indicates normal and 1 indicates an outlier.
        seed (int | None): Seed for random number generation used in
            splitting and sampling. None for true randomness.

    Returns
    -------
    Tuple[pandas.DataFrame, pandas.DataFrame, pandas.Series]:
        A tuple (x_train, x_test, y_test):

        - `x_train`: DataFrame of training features (normal samples only).
        - `x_test`: DataFrame of test features.
        - `y_test`: Series of test labels (0 for normal, 1 for outlier).
    """
    normal = df[df["Class"] == 0]
    n_train = len(normal) // 2
    n_test = min(1000, n_train // 3)
    n_test_outlier = n_test // 10
    n_test_normal = n_test - n_test_outlier

    x_train_full, test_set_normal_pool = train_test_split(
        normal, train_size=n_train, random_state=seed
    )
    x_train = x_train_full.drop(columns=["Class"])

    # Ensure enough samples are available for sampling
    # These checks could raise errors or adjust sample sizes if needed
    actual_n_test_normal = min(n_test_normal, len(test_set_normal_pool))
    test_normal = test_set_normal_pool.sample(n=actual_n_test_normal, random_state=seed)

    outliers_available = df[df["Class"] == 1]
    actual_n_test_outlier = min(n_test_outlier, len(outliers_available))
    test_outliers = outliers_available.sample(
        n=actual_n_test_outlier, random_state=seed
    )

    test_set = pd.concat([test_normal, test_outliers], ignore_index=True)

    x_test = test_set.drop(columns=["Class"])
    y_test = test_set["Class"]

    return x_train, x_test, y_test


def _cleanup_old_versions() -> None:
    """Remove cache directories from old dataset versions."""
    cache_dir = _get_cache_dir()
    cache_root = cache_dir.parent
    if not cache_root.exists():
        return

    current_version = DATASET_VERSION
    removed_count = 0

    for version_dir in cache_root.iterdir():
        if version_dir.is_dir() and version_dir.name != current_version:
            try:
                shutil.rmtree(version_dir)
                removed_count += 1
            except PermissionError:
                # On Windows, files may be locked by other processes (e.g., swap files)
                # Skip these directories to avoid test failures
                pass

    if removed_count > 0:
        logger.info("Cleaned up %d old dataset versions", removed_count)


def clear_cache(dataset: str | None = None, all_versions: bool = False) -> None:
    """Clear dataset cache.

    Args:
        dataset: Specific dataset name to clear (e.g., "breast").
                If None, clears all datasets for current version.
        all_versions: If True, clears cache for all dataset versions.
                     If False, only clears current version.
    """
    global _DATASET_CACHE

    if all_versions:
        # Clear entire cache directory (all versions)
        cache_root = _get_cache_dir().parent
        if cache_root.exists():
            try:
                shutil.rmtree(cache_root)
                logger.info("Cleared all dataset cache (all versions)")
            except PermissionError:
                # On Windows, files may be locked by other processes
                logger.warning("Could not clear all cache due to file permissions")
        _DATASET_CACHE.clear()
        return

    if dataset is not None:
        # Clear specific dataset
        filename = f"{dataset}{DATASET_SUFFIX}"

        # Remove from memory cache
        _DATASET_CACHE.pop(filename, None)

        # Remove from disk cache
        cache_dir = _get_cache_dir()
        cache_file = cache_dir / filename
        if cache_file.exists():
            cache_file.unlink()
            logger.info("Cleared cache for dataset: %s", dataset)
        else:
            logger.info("No cache found for dataset: %s", dataset)
    else:
        # Clear all datasets for current version
        cache_dir = _get_cache_dir()
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                logger.info("Cleared all dataset cache (v%s)", DATASET_VERSION)
            except PermissionError:
                # On Windows, files may be locked by other processes
                logger.warning(
                    "Could not clear cache directory (v%s) due to file permissions",
                    DATASET_VERSION,
                )
        _DATASET_CACHE.clear()


def list_cached_datasets() -> list[str]:
    """List all datasets cached (memory + disk).

    Returns
    -------
        List of cached dataset names (without .parquet.gz extension)
    """
    cached_names = set()

    # Add from memory cache
    cached_names.update(
        filename.removesuffix(DATASET_SUFFIX) for filename in _DATASET_CACHE.keys()
    )

    # Add from disk cache
    cache_dir = _get_cache_dir()
    if cache_dir.exists():
        cached_names.update(
            f.name.removesuffix(DATASET_SUFFIX)
            for f in cache_dir.glob(f"*{DATASET_SUFFIX}")
        )

    return sorted(cached_names)


def get_cache_info() -> dict:
    """Get comprehensive cache information.

    Returns
    -------
        Dictionary with memory and disk cache information
    """
    cache_dir = _get_cache_dir()
    cache_root = cache_dir.parent

    # Memory cache info
    memory_info = {
        "datasets": list(_DATASET_CACHE.keys()),
        "count": len(_DATASET_CACHE),
        "size_mb": round(
            sum(len(data) for data in _DATASET_CACHE.values()) / (1024 * 1024), 2
        ),
    }

    # Disk cache info
    disk_info = {
        "cache_dir": str(cache_dir),
        "current_version": DATASET_VERSION,
        "datasets": [],
        "total_size_mb": 0,
    }

    if cache_dir.exists():
        total_size = 0
        for cache_file in cache_dir.glob(f"*{DATASET_SUFFIX}"):
            size = cache_file.stat().st_size
            total_size += size
            size_mb = size / (1024 * 1024)
            # Use more precision for small files to avoid rounding to 0
            precision = 6 if size_mb < 0.01 else 2
            disk_info["datasets"].append(
                {
                    "name": cache_file.name.removesuffix(DATASET_SUFFIX),
                    "size_mb": round(size_mb, precision),
                }
            )
        disk_info["total_size_mb"] = round(total_size / (1024 * 1024), 2)

    # Check for old versions
    old_versions = []
    if cache_root.exists():
        for version_dir in cache_root.iterdir():
            if version_dir.is_dir() and version_dir.name != DATASET_VERSION:
                old_versions.append(version_dir.name)

    return {"memory": memory_info, "disk": disk_info, "old_versions": old_versions}


def get_cache_location() -> str:
    """Get the cache directory path."""
    return str(_get_cache_dir())
