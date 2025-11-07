"""Modern dataset loading module with DatasetManager architecture."""

import io
import os
import shutil
from collections import OrderedDict
from pathlib import Path
from urllib.error import URLError
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from nonconform.utils.data.registry import DATASET_REGISTRY, DatasetInfo
from nonconform.utils.func.enums import Dataset
from nonconform.utils.func.logger import get_logger

logger = get_logger("utils.data.load")


class DatasetManager:
    """Manages dataset loading, caching, and metadata."""

    def __init__(self) -> None:
        """Initialize the DatasetManager with configuration."""
        self.version: str = os.environ.get("UNQUAD_DATASET_VERSION", "v0.91.1-datasets")
        base_repo_url = (
            "https://github.com/OliverHennhoefer/nonconform/releases/download/"
        )
        self.base_url: str = os.environ.get(
            "UNQUAD_DATASET_URL",
            urljoin(base_repo_url, quote(self.version, safe="") + "/"),
        )
        self.suffix: str = ".npz"
        self._memory_cache: OrderedDict[str, bytes] = OrderedDict()
        self.max_cache_size: int = 16  # Limit memory cache to 16 datasets
        self._cache_dir: Path | None = None

    def _add_to_memory_cache(self, filename: str, data: bytes) -> None:
        """Add an item to the LRU memory cache and evict if over capacity."""
        self._memory_cache[filename] = data
        self._memory_cache.move_to_end(filename)
        if len(self._memory_cache) > self.max_cache_size:
            popped_filename, _ = self._memory_cache.popitem(last=False)
            logger.debug(f"Evicted {popped_filename} from memory cache")

    @property
    def cache_dir(self) -> Path:
        """Get cache directory, creating it lazily."""
        if self._cache_dir is None:
            self._cache_dir = (
                Path(
                    os.environ.get(
                        "UNQUAD_CACHE_DIR", Path.home() / ".cache" / "nonconform"
                    )
                )
                / self.version
            )
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        return self._cache_dir

    def load(
        self, dataset: Dataset, setup: bool = False, seed: int | None = None
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Load a dataset by enum value.

        Args:
            dataset: The dataset to load (use Dataset enum values).
            setup: If True, splits the data into training and testing sets
                   for anomaly detection tasks.
            seed: Random seed for data splitting if setup is True.

        Returns:
            If setup is False, returns the complete dataset as a DataFrame.
            If setup is True, returns a tuple: (x_train, x_test, y_test).

        Raises:
            ValueError: If the dataset is not found in the registry.
            URLError: If dataset download fails.
        """
        name = dataset.value  # Extract string value from enum

        if name not in DATASET_REGISTRY:
            available = ", ".join(sorted(DATASET_REGISTRY.keys()))
            raise ValueError(
                f"Dataset '{name}' not found. Available datasets: {available}"
            )

        filename = DATASET_REGISTRY[name].filename

        # Download or retrieve from cache
        data_bytes = self._download(filename)

        # Load NPZ file from bytes
        buffer = io.BytesIO(data_bytes)
        npz_file = np.load(buffer)

        # Extract data and labels
        data = npz_file["X"]
        labels = npz_file["y"]

        # Convert integer types to float32 for PyOD compatibility
        if data.dtype in [
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ]:
            data = data.astype(np.float32)

        # Create DataFrame with programmatic column names
        column_names = [f"V{i + 1}" for i in range(data.shape[1])]
        df = pd.DataFrame(data, columns=column_names)
        df["Class"] = labels

        if setup:
            return self._create_setup(df, seed)

        return df

    def _download(self, filename: str) -> bytes:
        """
        Download dataset with memory and disk caching.

        Args:
            filename: Name of the dataset file (e.g., "breast.npz").

        Returns:
            Bytes content of the dataset file.

        Raises:
            URLError: If download fails.
        """
        # Check memory cache first
        if filename in self._memory_cache:
            logger.debug(f"Loading {filename} from memory cache")
            self._memory_cache.move_to_end(filename)  # Mark as recently used
            return self._memory_cache[filename]

        # Check disk cache second
        cache_file = self.cache_dir / filename
        try:
            cache_exists = cache_file.exists()
        except PermissionError:
            logger.warning(
                "Skipping disk cache check for %s due to permission error", filename
            )
            cache_exists = False

        if cache_exists:
            try:
                logger.debug(f"Loading {filename} from disk cache (v{self.version})")
                with open(cache_file, "rb") as f:
                    data = f.read()
            except PermissionError:
                logger.warning(
                    "Unable to read %s from disk cache due to permission error",
                    filename,
                )
            else:
                self._add_to_memory_cache(filename, data)
                return data

        # Clean old versions before downloading
        self._cleanup_old_versions()

        # Download file
        url = urljoin(self.base_url, filename)
        logger.info(f"Downloading {filename} from {url}...")

        try:
            # Add headers to avoid GitHub rate limiting
            req = Request(url, headers={"User-Agent": "nonconform-dataset-loader"})

            with urlopen(req) as response:
                data = response.read()

        except URLError as e:
            raise URLError(f"Failed to download {filename}: {e!s}") from e

        # Cache in memory and on disk
        self._add_to_memory_cache(filename, data)
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "wb") as f:
                f.write(data)
        except PermissionError:
            logger.warning(
                "Unable to write %s to disk cache due to permission error", filename
            )

        logger.debug(f"Successfully cached {filename} ({len(data) / 1024:.1f} KB)")
        return data

    def _create_setup(
        self, df: pd.DataFrame, seed: int | None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Create an experimental train/test split from a dataset.

        This setup creates a scenario for anomaly detection where:
        - The training set contains only normal samples (Class 0).
        - The test set contains a mix of normal and anomaly samples.

        Args:
            df: The input DataFrame with a "Class" column.
            seed: Random seed for data splitting.

        Returns:
            A tuple (x_train, x_test, y_test).
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

        # Ensure enough samples are available
        actual_n_test_normal = min(n_test_normal, len(test_set_normal_pool))
        test_normal = test_set_normal_pool.sample(
            n=actual_n_test_normal, random_state=seed
        )

        outliers_available = df[df["Class"] == 1]
        actual_n_test_outlier = min(n_test_outlier, len(outliers_available))
        test_outliers = outliers_available.sample(
            n=actual_n_test_outlier, random_state=seed
        )

        test_set = pd.concat([test_normal, test_outliers], ignore_index=True)

        x_test = test_set.drop(columns=["Class"])
        y_test = test_set["Class"]

        return x_train, x_test, y_test

    def _cleanup_old_versions(self) -> None:
        """Remove cache directories from old dataset versions."""
        cache_root = self.cache_dir.parent
        if not cache_root.exists():
            return

        current_version = self.version
        removed_count = 0

        for version_dir in cache_root.iterdir():
            if version_dir.is_dir() and version_dir.name != current_version:
                try:
                    shutil.rmtree(version_dir)
                    removed_count += 1
                except PermissionError:
                    # Skip directories with permission issues
                    pass

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old dataset versions")

    def clear_cache(
        self, dataset: str | None = None, all_versions: bool = False
    ) -> None:
        """
        Clear dataset cache.

        Args:
            dataset: Specific dataset name to clear. If None, clears all.
            all_versions: If True, clears cache for all dataset versions.
        """
        if all_versions:
            # Clear entire cache directory (all versions)
            cache_root = self.cache_dir.parent
            if cache_root.exists():
                try:
                    shutil.rmtree(cache_root)
                    logger.info("Cleared all dataset cache (all versions)")
                except PermissionError:
                    logger.warning("Could not clear all cache due to file permissions")
            self._memory_cache.clear()
            return

        if dataset is not None:
            # Clear specific dataset
            filename = f"{dataset}{self.suffix}"

            # Remove from memory cache
            self._memory_cache.pop(filename, None)

            # Remove from disk cache
            cache_file = self.cache_dir / filename
            if cache_file.exists():
                cache_file.unlink()
                logger.info(f"Cleared cache for dataset: {dataset}")
            else:
                logger.info(f"No cache found for dataset: {dataset}")
        else:
            # Clear all datasets for current version
            if self.cache_dir.exists():
                try:
                    shutil.rmtree(self.cache_dir)
                    logger.info(f"Cleared all dataset cache (v{self.version})")
                except PermissionError:
                    logger.warning(
                        f"Could not clear cache directory (v{self.version}) "
                        f"due to file permissions"
                    )
            self._memory_cache.clear()

    def list_available(self) -> list[str]:
        """
        Get a list of all available dataset names.

        Returns:
            Sorted list of dataset names.
        """
        return sorted(DATASET_REGISTRY.keys())

    def get_info(self, dataset: Dataset) -> DatasetInfo:
        """
        Get metadata for a specific dataset.

        Args:
            dataset: The dataset to get info for (use Dataset enum values).

        Returns:
            DatasetInfo object with dataset metadata.

        Raises:
            ValueError: If the dataset is not found.
        """
        name = dataset.value  # Extract string value from enum

        if name not in DATASET_REGISTRY:
            available = ", ".join(sorted(DATASET_REGISTRY.keys()))
            raise ValueError(
                f"Dataset '{name}' not found. Available datasets: {available}"
            )
        return DATASET_REGISTRY[name]

    def get_cache_location(self) -> str:
        """
        Get the cache directory path.

        Returns:
            String path to the cache directory.
        """
        return str(self.cache_dir)

    @property
    def memory_cache_size(self) -> int:
        """Returns the number of datasets cached in memory.

        Returns:
            int: Number of datasets currently in memory cache.
        """
        return len(self._memory_cache)

    @property
    def is_cache_enabled(self) -> bool:
        """Returns whether disk caching is enabled.

        Returns:
            bool: True if cache directory exists and is writable.
        """
        try:
            return self.cache_dir.exists() and os.access(self.cache_dir, os.W_OK)
        except OSError:
            return False


# Create singleton instance
_manager = DatasetManager()


# Public API functions
def load(
    dataset: Dataset, setup: bool = False, seed: int | None = None
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load a benchmark anomaly detection dataset.

    Provides access to curated datasets commonly used for anomaly detection research.
    Datasets are automatically downloaded and cached locally for efficient reuse.

    Args:
        dataset: Dataset to load using Dataset enum (e.g., Dataset.SHUTTLE, ...).
        setup: If True, automatically splits data for anomaly detection workflow.
               Returns (x_train, x_test, y_test), x_train contains only normal samples.
        seed: Random seed for reproducible train/test splitting when setup=True.

    Returns:
        - If setup=False: Complete dataset as pd.DataFrame with 'label' column
        - If setup=True: Tuple of (x_train, x_test, y_test) where:
            - x_train: Normal samples for training (features only)
            - x_test: Mixed test samples (features only)
            - y_test: True labels for test samples (0=normal, 1=anomaly)

    Examples:
        Load complete dataset for exploration:
        ```python
        from nonconform.utils.data import load, Dataset

        # Load full dataset with labels
        df = load(Dataset.MAMMOGRAPHY)
        print(f"Dataset shape: {df.shape}")
        print(f"Anomaly rate: {df['label'].mean():.1%}")
        ```

        Load split data ready for conformal detection:
        ```python
        # Get training/test split for anomaly detection
        x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=42)

        # x_train contains only normal samples for detector training
        print(f"Training samples: {len(x_train)} (all normal)")
        print(f"Test samples: {len(x_test)} ({np.sum(y_test)} anomalies)")
        ```

    Available Datasets:
        Use `list_available()` to see all available datasets, or check enum values:
        Dataset.MAMMOGRAPHY, Dataset.SHUTTLE, Dataset.FRAUD, etc.
    """
    return _manager.load(dataset, setup=setup, seed=seed)


def list_available() -> list[str]:
    """
    Get a list of all available dataset names.

    Returns:
        Sorted list of dataset names.

    Examples:
        >>> datasets = list_available()
        >>> print(datasets)
        ['breast', 'fraud', 'ionosphere', ...]
    """
    return _manager.list_available()


def get_info(dataset: Dataset) -> DatasetInfo:
    """
    Get detailed metadata for a specific dataset.

    Args:
        dataset: The dataset to get info for (use Dataset enum values).

    Returns:
        DatasetInfo object with dataset metadata.

    Examples:
        >>> from nonconform.utils.data import Dataset
        >>> info = get_info(Dataset.BREAST)
        >>> print(info.description)
    """
    return _manager.get_info(dataset)


def clear_cache(dataset: str | None = None, all_versions: bool = False) -> None:
    """
    Clear dataset cache.

    Args:
        dataset: Specific dataset name to clear. If None, clears all.
        all_versions: If True, clears cache for all dataset versions.

    Examples:
        >>> clear_cache("breast")  # Clear specific dataset
        >>> clear_cache()  # Clear all datasets
        >>> clear_cache(all_versions=True)  # Clear all versions
    """
    _manager.clear_cache(dataset=dataset, all_versions=all_versions)


def get_cache_location() -> str:
    """
    Get the cache directory path.

    Returns:
        String path to the cache directory.

    Examples:
        >>> location = get_cache_location()
        >>> print(f"Cache stored at: {location}")
    """
    return _manager.get_cache_location()
