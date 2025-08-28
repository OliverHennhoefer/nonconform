import shutil
import tempfile
import unittest
from pathlib import Path
from urllib.error import URLError

from nonconform.utils.data import Dataset, clear_cache, get_cache_location, load
from nonconform.utils.data.load import _manager


class TestDatasetDownload(unittest.TestCase):
    """Test dataset caching functionality (memory + disk)."""

    def setUp(self):
        """Set up test environment."""
        # Create temporary cache root directory for testing
        self.temp_cache_root = Path(tempfile.mkdtemp())
        # The actual cache directory should be under temp_root/version_name/
        self.temp_cache_dir = self.temp_cache_root / _manager.version
        self.temp_cache_dir.mkdir(parents=True, exist_ok=True)

        # Store original cache_dir property
        self.original_cache_dir = _manager._cache_dir
        # Override cache directory for testing
        _manager._cache_dir = self.temp_cache_dir

        # Clear memory cache only
        _manager._memory_cache.clear()

    def tearDown(self):
        """Clean up after tests."""
        # Restore original cache directory
        _manager._cache_dir = self.original_cache_dir

        # Clear memory cache only
        _manager._memory_cache.clear()

        # Clean up temporary root directory (includes all subdirectories)
        if self.temp_cache_root.exists():
            try:
                shutil.rmtree(self.temp_cache_root)
            except PermissionError:
                # On Windows, may have permission issues with temp files
                pass

    def test_dataset_loading_and_caching(self):
        """Test that dataset loading works and uses caching."""
        dataset_filename = "breast_w.npz"  # Correct filename for breast dataset

        # 1. Load the dataset (should trigger download)
        try:
            df = load(Dataset.BREAST, setup=False)
            self.assertIsNotNone(df)
            self.assertGreater(len(df), 0)
        except URLError as e:
            self.skipTest(f"Network error, skipping test: {e}")
        except ImportError as e:
            self.skipTest(f"Missing dependencies, skipping test: {e}")

        # 2. Verify it's cached in memory
        self.assertIn(dataset_filename, _manager._memory_cache)
        self.assertGreater(len(_manager._memory_cache[dataset_filename]), 0)

        # 3. Verify it's cached on disk
        cache_file = self.temp_cache_dir / dataset_filename
        self.assertTrue(cache_file.exists())

        # 4. Clear specific dataset cache (need to use the correct filename without
        # extension)
        # The clear_cache function adds .npz extension automatically, but our dataset
        # has "breast_w.npz", so we need to call with "breast_w"
        clear_cache("breast_w")

        # 5. Verify it's gone from memory and disk
        self.assertNotIn(dataset_filename, _manager._memory_cache)
        self.assertFalse(cache_file.exists())

    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        # Load a dataset to populate cache
        try:
            load(Dataset.FRAUD, setup=False)
        except URLError as e:
            self.skipTest(f"Network error, skipping test: {e}")
        except ImportError as e:
            self.skipTest(f"Missing dependencies, skipping test: {e}")

        # Verify something is cached
        self.assertGreater(len(_manager._memory_cache), 0)

        # Clear all cache
        clear_cache()

        # Verify cache is empty
        self.assertEqual(len(_manager._memory_cache), 0)

    def test_cache_location(self):
        """Test get_cache_location function."""
        location = get_cache_location()
        self.assertIsInstance(location, str)
        self.assertTrue(len(location) > 0)


if __name__ == "__main__":
    unittest.main()
