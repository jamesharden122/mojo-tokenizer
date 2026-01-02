"""
Benchmark runner for tokenizer performance testing.

Measures encoding/decoding throughput and latency.
"""

from time import now


struct BenchmarkResult:
    """Result of a single benchmark run."""

    var name: String
    """Benchmark name."""

    var iterations: Int
    """Number of iterations run."""

    var total_time_ns: Int
    """Total time in nanoseconds."""

    var tokens_processed: Int
    """Total tokens processed."""

    var chars_processed: Int
    """Total characters processed."""

    fn __init__(
        out self,
        name: String,
        iterations: Int,
        total_time_ns: Int,
        tokens_processed: Int,
        chars_processed: Int
    ):
        """Create a benchmark result."""
        self.name = name
        self.iterations = iterations
        self.total_time_ns = total_time_ns
        self.tokens_processed = tokens_processed
        self.chars_processed = chars_processed

    fn avg_time_ns(self) -> Float64:
        """Average time per iteration in nanoseconds."""
        if self.iterations == 0:
            return 0.0
        return Float64(self.total_time_ns) / Float64(self.iterations)

    fn avg_time_ms(self) -> Float64:
        """Average time per iteration in milliseconds."""
        return self.avg_time_ns() / 1_000_000.0

    fn tokens_per_sec(self) -> Float64:
        """Tokens processed per second."""
        if self.total_time_ns == 0:
            return 0.0
        return Float64(self.tokens_processed) * 1_000_000_000.0 / Float64(self.total_time_ns)

    fn chars_per_sec(self) -> Float64:
        """Characters processed per second."""
        if self.total_time_ns == 0:
            return 0.0
        return Float64(self.chars_processed) * 1_000_000_000.0 / Float64(self.total_time_ns)

    fn print_summary(self):
        """Print benchmark summary."""
        print("Benchmark: " + self.name)
        print("  Iterations: " + str(self.iterations))
        print("  Total time: " + str(Float64(self.total_time_ns) / 1_000_000.0) + " ms")
        print("  Avg time: " + str(self.avg_time_ms()) + " ms")
        print("  Tokens/sec: " + str(Int(self.tokens_per_sec())))
        print("  Chars/sec: " + str(Int(self.chars_per_sec())))


struct BenchmarkRunner:
    """
    Runner for tokenizer benchmarks.

    Supports warmup iterations, multiple runs, and statistical analysis.
    """

    var warmup_iterations: Int
    """Number of warmup iterations (not counted)."""

    var iterations: Int
    """Number of measured iterations."""

    var _results: List[BenchmarkResult]
    """Collected results."""

    fn __init__(
        out self,
        iterations: Int = 100,
        warmup_iterations: Int = 10
    ):
        """Create a benchmark runner."""
        self.warmup_iterations = warmup_iterations
        self.iterations = iterations
        self._results = List[BenchmarkResult]()

    fn run_encode(
        mut self,
        name: String,
        tokenizer: BPETokenizer,
        text: String
    ) raises -> BenchmarkResult:
        """
        Benchmark encoding throughput.

        Args:
            name: Benchmark name.
            tokenizer: Tokenizer to benchmark.
            text: Text to encode.

        Returns:
            Benchmark result with timing statistics.
        """
        var tok = tokenizer  # Mutable copy

        # Warmup
        for _ in range(self.warmup_iterations):
            _ = tok.encode(text)

        # Clear cache stats after warmup
        tok.reset_cache_stats()

        # Timed runs
        var total_tokens = 0
        var start = now()

        for _ in range(self.iterations):
            var tokens = tok.encode(text)
            total_tokens += len(tokens)

        var end = now()
        var total_time_ns = end - start

        var result = BenchmarkResult(
            name,
            self.iterations,
            total_time_ns,
            total_tokens,
            len(text) * self.iterations
        )

        self._results.append(result)
        return result

    fn run_decode(
        mut self,
        name: String,
        tokenizer: BPETokenizer,
        tokens: List[Int]
    ) raises -> BenchmarkResult:
        """
        Benchmark decoding throughput.

        Args:
            name: Benchmark name.
            tokenizer: Tokenizer to benchmark.
            tokens: Tokens to decode.

        Returns:
            Benchmark result with timing statistics.
        """
        # Warmup
        for _ in range(self.warmup_iterations):
            _ = tokenizer.decode(tokens)

        # Timed runs
        var total_chars = 0
        var start = now()

        for _ in range(self.iterations):
            var text = tokenizer.decode(tokens)
            total_chars += len(text)

        var end = now()
        var total_time_ns = end - start

        var result = BenchmarkResult(
            name,
            self.iterations,
            total_time_ns,
            len(tokens) * self.iterations,
            total_chars
        )

        self._results.append(result)
        return result

    fn run_roundtrip(
        mut self,
        name: String,
        tokenizer: BPETokenizer,
        text: String
    ) raises -> BenchmarkResult:
        """
        Benchmark encode + decode roundtrip.

        Args:
            name: Benchmark name.
            tokenizer: Tokenizer to benchmark.
            text: Text to roundtrip.

        Returns:
            Benchmark result with timing statistics.
        """
        var tok = tokenizer

        # Warmup
        for _ in range(self.warmup_iterations):
            var tokens = tok.encode(text)
            _ = tok.decode(tokens)

        tok.reset_cache_stats()

        # Timed runs
        var total_tokens = 0
        var start = now()

        for _ in range(self.iterations):
            var tokens = tok.encode(text)
            _ = tok.decode(tokens)
            total_tokens += len(tokens)

        var end = now()
        var total_time_ns = end - start

        var result = BenchmarkResult(
            name + " (roundtrip)",
            self.iterations,
            total_time_ns,
            total_tokens,
            len(text) * self.iterations
        )

        self._results.append(result)
        return result

    fn results(self) -> List[BenchmarkResult]:
        """Get all collected results."""
        return self._results

    fn print_all_results(self):
        """Print all collected results."""
        print("\n=== Benchmark Results ===\n")
        for r in self._results:
            r[].print_summary()
            print("")


# Import for the benchmark functions
from ..bpe import BPETokenizer


fn run_benchmark(
    tokenizer: BPETokenizer,
    texts: List[String],
    iterations: Int = 100
) raises -> List[BenchmarkResult]:
    """
    Run standard benchmark suite on a tokenizer.

    Args:
        tokenizer: Tokenizer to benchmark.
        texts: List of texts to benchmark.
        iterations: Number of iterations per text.

    Returns:
        List of benchmark results.
    """
    var runner = BenchmarkRunner(iterations, 10)
    var results = List[BenchmarkResult]()

    for i in range(len(texts)):
        var text = texts[i]
        var name = "text_" + str(i) + " (" + str(len(text)) + " chars)"

        var encode_result = runner.run_encode(name + " encode", tokenizer, text)
        results.append(encode_result)

        # Get tokens for decode benchmark
        var tok = tokenizer
        var tokens = tok.encode(text)

        var decode_result = runner.run_decode(name + " decode", tokenizer, tokens)
        results.append(decode_result)

    return results
