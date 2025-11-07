from nonconform.utils.data import Dataset, clear_cache, get_cache_location


class TestMemoryCaching:
    def test_first_load_caches_in_memory(self, temp_manager, mock_urlopen):
        with mock_urlopen():
            temp_manager.load(Dataset.BREAST, setup=False)
            assert temp_manager.memory_cache_size == 1
            assert "breast_w.npz" in temp_manager._memory_cache

    def test_second_load_uses_memory_cache(self, temp_manager, mock_urlopen):
        with mock_urlopen() as mock:
            temp_manager.load(Dataset.BREAST, setup=False)
            call_count_after_first = mock.call_count
            temp_manager.load(Dataset.BREAST, setup=False)
            assert mock.call_count == call_count_after_first

    def test_memory_cache_lru_eviction_at_limit(self, temp_manager, mock_urlopen):
        dataset_mapping = {
            "breast": "breast_w.npz",
            "fraud": "fraud.npz",
            "ionosphere": "ionosphere.npz",
            "mammography": "mammography.npz",
            "musk": "musk.npz",
            "shuttle": "shuttle.npz",
            "thyroid": "thyroid.npz",
            "wbc": "wbc.npz",
            "annthyroid": "annthyroid.npz",
            "backdoor": "backdoor.npz",
            "cardio": "cardio.npz",
            "cover": "cover.npz",
            "donors": "donors.npz",
            "glass": "glass.npz",
            "hepatitis": "hepatitis.npz",
            "http": "http.npz",
        }

        with mock_urlopen():
            for i, dataset_name in enumerate(list(dataset_mapping.keys())[:16]):
                dataset_enum = Dataset[dataset_name.upper()]
                temp_manager.load(dataset_enum, setup=False)

            assert temp_manager.memory_cache_size == 16

            temp_manager.load(Dataset.LETTER, setup=False)
            assert temp_manager.memory_cache_size == 16
            assert "breast_w.npz" not in temp_manager._memory_cache

    def test_memory_cache_move_to_end_on_access(self, temp_manager, mock_urlopen):
        with mock_urlopen():
            temp_manager.load(Dataset.BREAST, setup=False)
            temp_manager.load(Dataset.FRAUD, setup=False)

            first_key = next(iter(temp_manager._memory_cache.keys()))
            assert first_key == "breast_w.npz"

            temp_manager.load(Dataset.BREAST, setup=False)
            last_key = list(temp_manager._memory_cache.keys())[-1]
            assert last_key == "breast_w.npz"


class TestDiskCaching:
    def test_first_load_creates_disk_cache(self, temp_manager, mock_urlopen):
        with mock_urlopen():
            temp_manager.load(Dataset.BREAST, setup=False)
            cache_file = temp_manager.cache_dir / "breast_w.npz"
            assert cache_file.exists()

    def test_disk_cache_persists_after_memory_clear(self, temp_manager, mock_urlopen):
        with mock_urlopen():
            temp_manager.load(Dataset.BREAST, setup=False)
            cache_file = temp_manager.cache_dir / "breast_w.npz"

            temp_manager._memory_cache.clear()
            assert cache_file.exists()

    def test_disk_cache_used_after_memory_clear(self, temp_manager, mock_urlopen):
        with mock_urlopen() as mock:
            temp_manager.load(Dataset.BREAST, setup=False)
            initial_calls = mock.call_count

            temp_manager._memory_cache.clear()
            temp_manager.load(Dataset.BREAST, setup=False)

            assert mock.call_count == initial_calls

    def test_disk_cache_populates_memory_cache(self, temp_manager, mock_urlopen):
        with mock_urlopen():
            temp_manager.load(Dataset.BREAST, setup=False)
            temp_manager._memory_cache.clear()

            temp_manager.load(Dataset.BREAST, setup=False)
            assert temp_manager.memory_cache_size == 1
            assert "breast_w.npz" in temp_manager._memory_cache


class TestCachePriority:
    def test_memory_before_disk(self, temp_manager, mock_urlopen):
        with mock_urlopen():
            temp_manager.load(Dataset.BREAST, setup=False)

        cache_file = temp_manager.cache_dir / "breast_w.npz"
        cache_file.unlink()

        df = temp_manager.load(Dataset.BREAST, setup=False)
        assert df is not None

    def test_disk_before_network(self, temp_manager, mock_urlopen):
        with mock_urlopen():
            temp_manager.load(Dataset.BREAST, setup=False)

        temp_manager._memory_cache.clear()

        with mock_urlopen() as mock:
            initial_calls = mock.call_count
            temp_manager.load(Dataset.BREAST, setup=False)
            assert mock.call_count == initial_calls


class TestVersionManagement:
    def test_cache_uses_version_specific_directory(self, temp_manager):
        cache_dir = temp_manager.cache_dir
        assert temp_manager.version in str(cache_dir)

    def test_cleanup_removes_old_versions(self, temp_manager, tmp_path):
        cache_root = temp_manager.cache_dir.parent
        old_version = cache_root / "v0.90.0"
        old_version.mkdir(parents=True, exist_ok=True)
        (old_version / "test.npz").write_bytes(b"fake data")

        temp_manager._cleanup_old_versions()

        assert not old_version.exists()
        assert temp_manager.cache_dir.exists()

    def test_cleanup_preserves_current_version(self, temp_manager, tmp_path):
        cache_root = temp_manager.cache_dir.parent
        current_file = temp_manager.cache_dir / "current.npz"
        current_file.write_bytes(b"current data")

        old_version = cache_root / "v0.85.0"
        old_version.mkdir(parents=True, exist_ok=True)

        temp_manager._cleanup_old_versions()

        assert current_file.exists()
        assert not old_version.exists()

    def test_cleanup_removes_multiple_old_versions(self, temp_manager):
        cache_root = temp_manager.cache_dir.parent
        old_versions = ["v0.90.0", "v0.85.0", "v0.80.0"]

        for version in old_versions:
            version_dir = cache_root / version
            version_dir.mkdir(parents=True, exist_ok=True)

        temp_manager._cleanup_old_versions()

        for version in old_versions:
            assert not (cache_root / version).exists()


class TestCacheClearing:
    def test_clear_specific_dataset_from_memory(self, temp_manager, mock_urlopen):
        with mock_urlopen():
            temp_manager.load(Dataset.BREAST, setup=False)

        temp_manager.clear_cache("breast_w")
        assert "breast_w.npz" not in temp_manager._memory_cache

    def test_clear_specific_dataset_from_disk(self, temp_manager, mock_urlopen):
        with mock_urlopen():
            temp_manager.load(Dataset.BREAST, setup=False)

        temp_manager.clear_cache("breast_w")
        cache_file = temp_manager.cache_dir / "breast_w.npz"
        assert not cache_file.exists()

    def test_clear_all_datasets_memory(self, temp_manager, mock_urlopen):
        with mock_urlopen():
            temp_manager.load(Dataset.BREAST, setup=False)
            temp_manager.load(Dataset.FRAUD, setup=False)

        temp_manager.clear_cache()
        assert temp_manager.memory_cache_size == 0

    def test_clear_all_datasets_disk(self, temp_manager, mock_urlopen):
        with mock_urlopen():
            temp_manager.load(Dataset.BREAST, setup=False)

        temp_manager.clear_cache()
        assert not temp_manager.cache_dir.exists()

    def test_clear_all_versions(self, temp_manager, mock_urlopen, tmp_path):
        cache_root = temp_manager.cache_dir.parent

        with mock_urlopen():
            temp_manager.load(Dataset.BREAST, setup=False)

        old_version = cache_root / "v0.90.0"
        old_version.mkdir(parents=True, exist_ok=True)

        temp_manager.clear_cache(all_versions=True)

        assert not cache_root.exists()
        assert temp_manager.memory_cache_size == 0

    def test_clear_nonexistent_dataset(self, temp_manager):
        temp_manager.clear_cache("nonexistent")


class TestCacheProperties:
    def test_memory_cache_size_empty(self, temp_manager):
        assert temp_manager.memory_cache_size == 0

    def test_memory_cache_size_after_loads(self, temp_manager, mock_urlopen):
        with mock_urlopen():
            temp_manager.load(Dataset.BREAST, setup=False)
            assert temp_manager.memory_cache_size == 1

            temp_manager.load(Dataset.FRAUD, setup=False)
            assert temp_manager.memory_cache_size == 2

    def test_is_cache_enabled_true(self, temp_manager):
        assert temp_manager.is_cache_enabled is True

    def test_get_cache_location_returns_string(self, temp_manager):
        location = temp_manager.get_cache_location()
        assert isinstance(location, str)
        assert len(location) > 0

    def test_cache_location_contains_version(self, temp_manager):
        location = temp_manager.get_cache_location()
        assert temp_manager.version in location


class TestPublicCacheFunctions:
    def test_clear_cache_function(self, mock_urlopen):
        from nonconform.utils.data.load import _manager

        with mock_urlopen():
            _manager.load(Dataset.BREAST, setup=False)

        initial_size = _manager.memory_cache_size
        clear_cache("breast_w")
        assert _manager.memory_cache_size < initial_size or initial_size == 0

    def test_get_cache_location_function(self):
        location = get_cache_location()
        assert isinstance(location, str)
        assert len(location) > 0
