"""
Benchmark utilities for mojo-tokenizer.

Performance measurement and comparison tools.
"""

from .runner import BenchmarkRunner, BenchmarkResult, run_benchmark
from .stats import BenchmarkStats, calculate_stats
