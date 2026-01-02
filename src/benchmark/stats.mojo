"""
Statistical utilities for benchmark analysis.
"""

from math import sqrt


struct BenchmarkStats:
    """Statistical summary of benchmark results."""

    var name: String
    """Benchmark name."""

    var count: Int
    """Number of samples."""

    var mean: Float64
    """Mean value."""

    var std_dev: Float64
    """Standard deviation."""

    var min_val: Float64
    """Minimum value."""

    var max_val: Float64
    """Maximum value."""

    var median: Float64
    """Median value."""

    var p95: Float64
    """95th percentile."""

    var p99: Float64
    """99th percentile."""

    fn __init__(out self, name: String):
        """Create empty stats."""
        self.name = name
        self.count = 0
        self.mean = 0.0
        self.std_dev = 0.0
        self.min_val = 0.0
        self.max_val = 0.0
        self.median = 0.0
        self.p95 = 0.0
        self.p99 = 0.0

    fn print_summary(self):
        """Print statistical summary."""
        print("Statistics: " + self.name)
        print("  Count: " + str(self.count))
        print("  Mean: " + str(self.mean))
        print("  Std Dev: " + str(self.std_dev))
        print("  Min: " + str(self.min_val))
        print("  Max: " + str(self.max_val))
        print("  Median: " + str(self.median))
        print("  P95: " + str(self.p95))
        print("  P99: " + str(self.p99))


fn calculate_stats(name: String, values: List[Float64]) -> BenchmarkStats:
    """
    Calculate statistics for a list of values.

    Args:
        name: Name for the statistics.
        values: List of values to analyze.

    Returns:
        BenchmarkStats with calculated values.
    """
    var stats = BenchmarkStats(name)

    if len(values) == 0:
        return stats

    stats.count = len(values)

    # Calculate mean
    var sum_val: Float64 = 0.0
    for v in values:
        sum_val += v[]
    stats.mean = sum_val / Float64(len(values))

    # Calculate std dev
    var sum_sq: Float64 = 0.0
    for v in values:
        var diff = v[] - stats.mean
        sum_sq += diff * diff
    stats.std_dev = sqrt(sum_sq / Float64(len(values)))

    # Find min and max
    stats.min_val = values[0]
    stats.max_val = values[0]
    for v in values:
        if v[] < stats.min_val:
            stats.min_val = v[]
        if v[] > stats.max_val:
            stats.max_val = v[]

    # Sort for percentiles (simple bubble sort for now)
    var sorted_vals = List[Float64]()
    for v in values:
        sorted_vals.append(v[])

    for i in range(len(sorted_vals)):
        for j in range(len(sorted_vals) - 1 - i):
            if sorted_vals[j] > sorted_vals[j + 1]:
                var temp = sorted_vals[j]
                sorted_vals[j] = sorted_vals[j + 1]
                sorted_vals[j + 1] = temp

    # Calculate percentiles
    var n = len(sorted_vals)
    stats.median = sorted_vals[n // 2]
    stats.p95 = sorted_vals[Int(Float64(n) * 0.95)]
    stats.p99 = sorted_vals[Int(Float64(n) * 0.99)]

    return stats


fn compare_benchmarks(
    baseline: BenchmarkStats,
    comparison: BenchmarkStats
) -> Float64:
    """
    Compare two benchmarks and return speedup factor.

    Args:
        baseline: Baseline benchmark stats.
        comparison: Comparison benchmark stats.

    Returns:
        Speedup factor (>1 means comparison is faster).
    """
    if comparison.mean == 0.0:
        return 0.0
    return baseline.mean / comparison.mean


fn format_throughput(tokens_per_sec: Float64) -> String:
    """Format throughput as a human-readable string."""
    if tokens_per_sec >= 1_000_000:
        return str(tokens_per_sec / 1_000_000.0) + "M tokens/sec"
    elif tokens_per_sec >= 1_000:
        return str(tokens_per_sec / 1_000.0) + "K tokens/sec"
    else:
        return str(tokens_per_sec) + " tokens/sec"


fn format_latency(ns: Float64) -> String:
    """Format latency as a human-readable string."""
    if ns >= 1_000_000_000:
        return str(ns / 1_000_000_000.0) + " s"
    elif ns >= 1_000_000:
        return str(ns / 1_000_000.0) + " ms"
    elif ns >= 1_000:
        return str(ns / 1_000.0) + " us"
    else:
        return str(ns) + " ns"
