"""
End-to-end BPE tokenizer benchmark.

Run from project root (after building package):
    mojo package src -o mojo_tokenizer.mojopkg
    mojo run benchmarks/bench_full_encode.mojo
"""

from time import perf_counter_ns
from sys import argv

from mojo_tokenizer.bpe import BPETokenizer


fn read_file(path: String) raises -> String:
    """Read entire file contents."""
    with open(path, "r") as f:
        return f.read()


fn main() raises:
    print("=== mojo-tokenizer v0.4.0 Phase 1 Benchmark ===\n")

    var vocab_path = "benchmarks/data/llama3.tiktoken"
    var text_path = "benchmarks/data/sherlock.txt"

    if len(argv()) > 1:
        vocab_path = argv()[1]
    if len(argv()) > 2:
        text_path = argv()[2]

    # Load tokenizer
    print("Loading tokenizer from:", vocab_path)
    var start = perf_counter_ns()
    var tokenizer = BPETokenizer.from_tiktoken(vocab_path)
    var load_time = (perf_counter_ns() - start) // 1_000_000
    print("Load time:", load_time, "ms")
    print("Vocab size:", tokenizer.vocab_size())
    print()

    # Load text
    print("Loading text from:", text_path)
    var text = read_file(text_path)
    var text_bytes = len(text)
    print("Text size:", text_bytes, "bytes (", text_bytes // 1024, "KB)")
    print()

    # Warmup run
    print("Warmup run...")
    tokenizer.clear_cache()
    _ = tokenizer.encode(text[:1000])
    tokenizer.clear_cache()

    # Benchmark encoding
    print("\n--- Benchmark Results ---")

    # Single large file encode (cold cache)
    print("\n1. Single file encoding (cold cache):")
    tokenizer.clear_cache()
    start = perf_counter_ns()
    var tokens = tokenizer.encode(text)
    var encode_time_ns = perf_counter_ns() - start
    var encode_time_ms = Float64(encode_time_ns) / 1_000_000.0
    var token_count = len(tokens)
    var throughput = Float64(token_count) / (encode_time_ms / 1000.0)

    print("   Tokens:", token_count)
    print("   Time:", Int(encode_time_ms), "ms")
    print("   Throughput:", Int(throughput / 1000), "K tok/s")
    var stats = tokenizer.cache_stats()
    print("   Cache hits:", stats[0], "misses:", stats[1])
    print("   Hit rate:", Int(tokenizer.cache_hit_rate() * 100), "%")

    # Repeat encode (warm cache)
    print("\n2. Repeat encoding (warm cache):")
    tokenizer.reset_cache_stats()
    start = perf_counter_ns()
    var tokens2 = tokenizer.encode(text)
    var warm_time_ns = perf_counter_ns() - start
    var warm_time_ms = Float64(warm_time_ns) / 1_000_000.0
    var warm_throughput = Float64(len(tokens2)) / (warm_time_ms / 1000.0)

    print("   Tokens:", len(tokens2))
    print("   Time:", Int(warm_time_ms), "ms")
    print("   Throughput:", Int(warm_throughput / 1000), "K tok/s")
    stats = tokenizer.cache_stats()
    print("   Cache hits:", stats[0], "misses:", stats[1])
    print("   Hit rate:", Int(tokenizer.cache_hit_rate() * 100), "%")

    # Multiple iterations (average)
    print("\n3. Average over 5 iterations (warm cache):")
    var total_time: Int64 = 0
    var total_tokens = 0
    for i in range(5):
        tokenizer.reset_cache_stats()
        start = perf_counter_ns()
        var t = tokenizer.encode(text)
        total_time += perf_counter_ns() - start
        total_tokens += len(t)

    var avg_time_ms = Float64(total_time) / 5.0 / 1_000_000.0
    var avg_throughput = Float64(total_tokens) / 5.0 / (avg_time_ms / 1000.0)
    print("   Avg time:", Int(avg_time_ms), "ms")
    print("   Avg throughput:", Int(avg_throughput / 1000), "K tok/s")

    # Summary
    print("\n=== Summary ===")
    print("Text:", text_bytes // 1024, "KB")
    print("Tokens:", token_count)
    print("Cold cache:", Int(throughput / 1000), "K tok/s")
    print("Warm cache:", Int(warm_throughput / 1000), "K tok/s")
    print("5-run avg:", Int(avg_throughput / 1000), "K tok/s")

    # Performance target check
    var target = 4000  # Phase 1 target: 4M tok/s
    if Int(avg_throughput / 1000) >= target:
        print("\n✓ Phase 1 target ACHIEVED:", Int(avg_throughput / 1000), "K tok/s >= 4M tok/s")
    else:
        print("\n✗ Phase 1 target NOT met:", Int(avg_throughput / 1000), "K tok/s < 4M tok/s")
