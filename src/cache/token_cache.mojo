"""
Token cache for mojo-tokenizer.

Caches tokenization results for common words to avoid
recomputing BPE merges. Uses simple LRU eviction.

Based on performance best practices:
- Pre-sized collections
- Move semantics where possible
"""


struct TokenCache(Movable):
    """
    LRU cache for tokenized words.

    Caches word -> token IDs mapping to avoid recomputing
    BPE merges for common words. 80%+ hit rate is typical
    for natural language text.
    """

    var _cache: Dict[String, List[Int]]
    var _access_order: List[String]  # For LRU eviction
    var _capacity: Int
    var _hits: Int
    var _misses: Int

    fn __init__(out self, capacity: Int = 10000):
        """
        Create a token cache with given capacity.

        Args:
            capacity: Maximum number of entries to cache.
                      Default 10000 covers most common words.
        """
        self._cache = Dict[String, List[Int]]()
        self._access_order = List[String]()
        self._capacity = capacity
        self._hits = 0
        self._misses = 0

    fn __moveinit__(out self, deinit existing: Self):
        """Move constructor."""
        self._cache = existing._cache^
        self._access_order = existing._access_order^
        self._capacity = existing._capacity
        self._hits = existing._hits
        self._misses = existing._misses

    fn get(mut self, key: String) -> Optional[List[Int]]:
        """
        Get cached token IDs for a word.

        Args:
            key: The word to look up.

        Returns:
            Optional list of token IDs, or None if not cached.
        """
        if key in self._cache:
            self._hits += 1
            self._move_to_front(key)
            try:
                return self._cache[key].copy()
            except:
                return None
        self._misses += 1
        return None

    fn put(mut self, key: String, var value: List[Int]):
        """
        Cache token IDs for a word.

        Uses move semantics to avoid copying the value.

        Args:
            key: The word to cache.
            value: The token IDs (moved, not copied).
        """
        # Evict if at capacity
        if len(self._cache) >= self._capacity and key not in self._cache:
            self._evict_lru()

        self._cache[key] = value^
        self._access_order.append(key)

    fn contains(self, key: String) -> Bool:
        """Check if key is in cache."""
        return key in self._cache

    fn hit_rate(self) -> Float64:
        """
        Get cache hit rate.

        Returns:
            Hit rate as fraction (0.0 to 1.0).
        """
        var total = self._hits + self._misses
        if total == 0:
            return 0.0
        return Float64(self._hits) / Float64(total)

    fn hits(self) -> Int:
        """Get total cache hits."""
        return self._hits

    fn misses(self) -> Int:
        """Get total cache misses."""
        return self._misses

    fn size(self) -> Int:
        """Get current cache size."""
        return len(self._cache)

    fn capacity(self) -> Int:
        """Get cache capacity."""
        return self._capacity

    fn clear(mut self):
        """Clear all cached entries."""
        self._cache = Dict[String, List[Int]]()
        self._access_order = List[String]()
        # Don't reset hit/miss counts

    fn reset_stats(mut self):
        """Reset hit/miss statistics."""
        self._hits = 0
        self._misses = 0

    fn _move_to_front(mut self, key: String):
        """Move key to front of access order (most recently used)."""
        # Simple approach: don't actually reorder, just append
        # Real LRU would remove and re-add, but this is faster
        # and good enough for our purposes
        self._access_order.append(key)

        # Compact if access_order grows too large
        if len(self._access_order) > self._capacity * 2:
            self._compact_access_order()

    fn _evict_lru(mut self):
        """Evict least recently used entry."""
        if len(self._access_order) == 0:
            return

        # Find first entry in access_order that's still in cache
        var evicted = False
        while len(self._access_order) > 0 and not evicted:
            var oldest = self._access_order[0]

            # Remove from front
            var new_order = List[String]()
            for i in range(1, len(self._access_order)):
                new_order.append(self._access_order[i])
            self._access_order = new_order^

            # Check if still in cache (might have been evicted earlier)
            if oldest in self._cache:
                # Count remaining references in access_order
                var count = 0
                for ki in range(len(self._access_order)):
                    if self._access_order[ki] == oldest:
                        count += 1
                        break

                # Only evict if no other references
                if count == 0:
                    # Remove from cache
                    var new_cache = Dict[String, List[Int]]()
                    for item in self._cache.items():
                        if item.key != oldest:
                            try:
                                # Use item.value directly to avoid aliasing
                                new_cache[item.key] = item.value.copy()
                            except:
                                pass
                    self._cache = new_cache^
                    evicted = True

    fn _compact_access_order(mut self):
        """Remove duplicate entries in access order, keeping last occurrence."""
        var seen = Dict[String, Bool]()
        var new_order = List[String]()

        # Iterate backwards, keeping first occurrence (which is last in original)
        for i in range(len(self._access_order) - 1, -1, -1):
            var key = self._access_order[i]
            if key not in seen and key in self._cache:
                seen[key] = True

        # Now iterate forward, adding only if in seen
        var added = Dict[String, Bool]()
        for i in range(len(self._access_order)):
            var key = self._access_order[i]
            if key in seen and key not in added:
                new_order.append(key)
                added[key] = True

        self._access_order = new_order^


struct MergeCache(Movable):
    """
    Cache for BPE merge rule lookups.

    Uses hash-based lookup for O(1) merge rank queries.
    """

    var _ranks: Dict[UInt64, Int]  # hash(pair) -> rank
    var _size: Int

    fn __init__(out self):
        self._ranks = Dict[UInt64, Int]()
        self._size = 0

    fn __moveinit__(out self, deinit existing: Self):
        """Move constructor."""
        self._ranks = existing._ranks^
        self._size = existing._size

    fn add(mut self, first: String, second: String, rank: Int):
        """Add a merge rule."""
        var hash = self._hash_pair(first, second)
        self._ranks[hash] = rank
        self._size += 1

    fn get_rank(self, first: String, second: String) -> Int:
        """
        Get merge rank for a token pair.

        Returns:
            Rank (lower = higher priority), or -1 if not found.
        """
        var hash = self._hash_pair(first, second)
        if hash in self._ranks:
            try:
                return self._ranks[hash]
            except:
                return -1
        return -1

    fn has_merge(self, first: String, second: String) -> Bool:
        """Check if merge rule exists for pair."""
        var hash = self._hash_pair(first, second)
        return hash in self._ranks

    fn size(self) -> Int:
        """Get number of merge rules."""
        return self._size

    @always_inline
    fn _hash_pair(self, a: String, b: String) -> UInt64:
        """
        Fast FNV-1a hash for token pair.

        Uses separator byte to ensure hash("ab", "c") != hash("a", "bc").
        """
        var hash: UInt64 = 14695981039346656037  # FNV offset basis
        alias FNV_PRIME: UInt64 = 1099511628211

        for i in range(len(a)):
            hash ^= UInt64(ord(a[i]))
            hash *= FNV_PRIME

        hash ^= UInt64(0xFF)  # Separator
        hash *= FNV_PRIME

        for i in range(len(b)):
            hash ^= UInt64(ord(b[i]))
            hash *= FNV_PRIME

        return hash
