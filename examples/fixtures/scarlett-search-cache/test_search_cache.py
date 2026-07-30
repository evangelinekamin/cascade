"""Tests for the in-memory LRU+TTL search cache."""

from unittest.mock import patch


from services.search_cache import SearchCache, normalize_cache_key


class TestSearchCache:
    def test_put_and_get(self):
        cache = SearchCache(max_entries=10, default_ttl_hours=1)
        cache.put("key1", [{"name": "Dress", "price": 29.99}])
        result = cache.get("key1")
        assert result == [{"name": "Dress", "price": 29.99}]

    def test_cache_miss_returns_none(self):
        cache = SearchCache()
        assert cache.get("nonexistent") is None

    def test_expired_entry_returns_none(self):
        cache = SearchCache(default_ttl_hours=1)
        cache.put("key1", [{"name": "Dress"}])

        # Simulate time passing beyond TTL
        with patch("services.search_cache.time") as mock_time:
            # Initial put uses real time; mock the get to be past expiry
            mock_time.monotonic.return_value = 1e12  # far in the future
            result = cache.get("key1")
            assert result is None

    def test_lru_eviction(self):
        cache = SearchCache(max_entries=2, default_ttl_hours=1)
        cache.put("key1", [{"name": "A"}])
        cache.put("key2", [{"name": "B"}])
        cache.put("key3", [{"name": "C"}])

        # key1 should have been evicted (oldest)
        assert cache.get("key1") is None
        assert cache.get("key2") == [{"name": "B"}]
        assert cache.get("key3") == [{"name": "C"}]

    def test_lru_eviction_respects_access_order(self):
        cache = SearchCache(max_entries=2, default_ttl_hours=1)
        cache.put("key1", [{"name": "A"}])
        cache.put("key2", [{"name": "B"}])

        # Access key1 to make it most-recently used
        cache.get("key1")

        # Now insert key3 — key2 should be evicted (least recently used)
        cache.put("key3", [{"name": "C"}])

        assert cache.get("key1") == [{"name": "A"}]
        assert cache.get("key2") is None
        assert cache.get("key3") == [{"name": "C"}]

    def test_stats_tracking(self):
        cache = SearchCache(max_entries=2, default_ttl_hours=1)
        cache.put("key1", [{"name": "A"}])

        cache.get("key1")  # hit
        cache.get("key2")  # miss
        cache.get("key1")  # hit
        cache.get("key3")  # miss

        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 2
        assert stats["size"] == 1
        assert stats["hit_rate"] == 0.5

    def test_eviction_counted_in_stats(self):
        cache = SearchCache(max_entries=1, default_ttl_hours=1)
        cache.put("key1", [{"name": "A"}])
        cache.put("key2", [{"name": "B"}])

        assert cache.stats()["evictions"] == 1

    def test_invalidate_removes_entry(self):
        cache = SearchCache()
        cache.put("key1", [{"name": "A"}])
        assert cache.get("key1") is not None

        cache.invalidate("key1")
        assert cache.get("key1") is None

    def test_invalidate_nonexistent_key_is_noop(self):
        cache = SearchCache()
        cache.invalidate("nonexistent")  # should not raise

    def test_clear_empties_cache(self):
        cache = SearchCache()
        cache.put("key1", [{"name": "A"}])
        cache.put("key2", [{"name": "B"}])
        assert cache.stats()["size"] == 2

        cache.clear()
        assert cache.stats()["size"] == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_put_overwrites_existing(self):
        cache = SearchCache()
        cache.put("key1", [{"name": "A"}])
        cache.put("key1", [{"name": "B"}])
        assert cache.get("key1") == [{"name": "B"}]
        assert cache.stats()["size"] == 1

    def test_custom_ttl_per_entry(self):
        cache = SearchCache(default_ttl_hours=1)
        cache.put("key1", [{"name": "A"}], ttl_hours=0)  # expires immediately

        with patch("services.search_cache.time") as mock_time:
            mock_time.monotonic.return_value = 1e12
            assert cache.get("key1") is None


class TestNormalizeCacheKey:
    def test_basic_key(self):
        key = normalize_cache_key("blue dress", "amazon", "dress")
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex

    def test_case_insensitive(self):
        k1 = normalize_cache_key("Blue Dress", "Amazon", "DRESS")
        k2 = normalize_cache_key("blue dress", "amazon", "dress")
        assert k1 == k2

    def test_whitespace_collapsed(self):
        k1 = normalize_cache_key("blue   dress", "amazon")
        k2 = normalize_cache_key("blue dress", "amazon")
        assert k1 == k2

    def test_filter_order_independent(self):
        k1 = normalize_cache_key(
            "dress", "amazon", filters={"color": "blue", "size": "M"}
        )
        k2 = normalize_cache_key(
            "dress", "amazon", filters={"size": "M", "color": "blue"}
        )
        assert k1 == k2

    def test_different_queries_produce_different_keys(self):
        k1 = normalize_cache_key("blue dress", "amazon")
        k2 = normalize_cache_key("red skirt", "amazon")
        assert k1 != k2

    def test_different_retailers_produce_different_keys(self):
        k1 = normalize_cache_key("blue dress", "amazon")
        k2 = normalize_cache_key("blue dress", "uniqlo")
        assert k1 != k2

    def test_none_category_handled(self):
        k1 = normalize_cache_key("dress", "amazon", None)
        k2 = normalize_cache_key("dress", "amazon", "")
        assert k1 == k2

    def test_none_filters_handled(self):
        key = normalize_cache_key("dress", "amazon", filters=None)
        assert isinstance(key, str)
