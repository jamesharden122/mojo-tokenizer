"""Declared byte-range patterns that emit canonical vocabulary token IDs."""

from std.collections.interval import Interval


struct BytePattern(Copyable, Movable):
    """A fixed-length sequence of deterministic byte intervals."""

    var intervals: List[Interval[Int]]
    var token_id: Int

    def __init__(
        out self,
        intervals: List[Interval[Int]],
        token_id: Int,
    ):
        self.intervals = intervals.copy()
        self.token_id = token_id


struct BytePatternSet(Copyable, Movable):
    """Pattern definitions used to build the tokenizer's ranged trie."""

    var _patterns: List[BytePattern]

    def __init__(out self):
        self._patterns = List[BytePattern]()

    def add(
        mut self,
        intervals: List[Interval[Int]],
        token_id: Int,
    ) raises:
        if len(intervals) == 0:
            raise Error("Byte pattern cannot be empty")
        if token_id < 0:
            raise Error("Byte pattern token ID must be nonnegative")
        for i in range(len(intervals)):
            var interval = intervals[i]
            if interval.start < 0 or interval.end > 256 or interval.start >= interval.end:
                raise Error("Byte pattern intervals must be nonempty and inside [0, 256)")
        for pattern_index in range(len(self._patterns)):
            var existing = self._patterns[pattern_index].copy()
            var common_length = min(len(intervals), len(existing.intervals))
            var shares_path = True
            for interval_index in range(common_length):
                var left = intervals[interval_index]
                var right = existing.intervals[interval_index]
                if left.start == right.start and left.end == right.end:
                    continue
                if left.overlaps(right):
                    raise Error("Byte pattern sibling intervals cannot overlap")
                shares_path = False
                break
            if shares_path and len(intervals) == len(existing.intervals):
                raise Error("Byte pattern path is already registered")
        self._patterns.append(BytePattern(intervals, token_id))

    def size(self) -> Int:
        return len(self._patterns)

    def get(self, index: Int) -> BytePattern:
        return self._patterns[index].copy()
