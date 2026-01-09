"""
HuggingFace BPE Validation - Mojo Side

Tests mojo-tokenizer against HuggingFace tokenizer.json files.

Run:
    mojo run test_huggingface.mojo
"""

from time import perf_counter_ns
from src.bpe import BPETokenizer


fn read_file(path: String) raises -> String:
    """Read entire file contents."""
    with open(path, "r") as f:
        return f.read()


fn print_tokens(tokens: List[Int]):
    """Print a list of tokens."""
    for i in range(len(tokens)):
        print(" ", tokens[i], end="")
    print()


fn tokens_match(actual: List[Int], expected: List[Int]) -> Bool:
    """Check if two token lists match."""
    if len(actual) != len(expected):
        return False
    for i in range(len(expected)):
        if actual[i] != expected[i]:
            return False
    return True


fn benchmark_encode(
    mut tokenizer: BPETokenizer,
    text: String,
    iterations: Int = 20,
    warmup: Int = 3
) raises -> Tuple[Float64, Int]:
    """Benchmark encoding performance."""
    # Warmup
    var token_count = 0
    for _ in range(warmup):
        var tokens = tokenizer.encode(text)
        token_count = len(tokens)

    # Timed runs
    var start = perf_counter_ns()
    for _ in range(iterations):
        var tokens = tokenizer.encode(text)
        _ = len(tokens)
    var elapsed_ns = perf_counter_ns() - start

    var elapsed_sec = Float64(elapsed_ns) / 1e9
    var tok_per_sec = Float64(token_count * iterations) / elapsed_sec

    return (tok_per_sec, token_count)


fn test_qwen() raises:
    """Test Qwen 2 tokenizer."""
    print("=" * 70)
    print("Testing: Qwen 2 (Qwen/Qwen2-1.5B)")
    print("=" * 70)

    var tokenizer_path = "data/huggingface/Qwen_Qwen2-1.5B/tokenizer.json"

    print("  Loading tokenizer.json...")
    var start = perf_counter_ns()
    var tokenizer = BPETokenizer.from_huggingface(tokenizer_path)
    var load_time = Float64(perf_counter_ns() - start) / 1e6

    print("  Vocab size:", tokenizer.vocab_size())
    print("  Load time:", load_time, "ms")

    print("\n--- Correctness Test ---")

    # Test "Hello, world!" -> [9707, 11, 1879, 0]
    var text1 = "Hello, world!"
    var tokens1 = tokenizer.encode(text1)
    var expected1 = List[Int](9707, 11, 1879, 0)
    var matches1 = tokens_match(tokens1, expected1)

    print("  'Hello, world!' ->", "✓" if matches1 else "✗")
    if not matches1:
        print("    Expected:", end="")
        print_tokens(expected1)
        print("    Actual:  ", end="")
        print_tokens(tokens1)

    # Test "def main():" -> [750, 1887, 4555]
    var text2 = "def main():"
    var tokens2 = tokenizer.encode(text2)
    var expected2 = List[Int](750, 1887, 4555)
    var matches2 = tokens_match(tokens2, expected2)

    print("  'def main():' ->", "✓" if matches2 else "✗")
    if not matches2:
        print("    Expected:", end="")
        print_tokens(expected2)
        print("    Actual:  ", end="")
        print_tokens(tokens2)

    var passed = (1 if matches1 else 0) + (1 if matches2 else 0)
    print("\n  Passed:", passed, "/ 2")

    # Performance benchmark
    print("\n--- Performance Benchmark ---")
    var sherlock = read_file("data/sherlock.txt")
    print("  Using: sherlock.txt (", len(sherlock), "chars)")

    var result = benchmark_encode(tokenizer, sherlock, iterations=20, warmup=3)
    var tok_per_sec = result[0]
    var token_count = result[1]

    print("  Encoding:", tok_per_sec / 1e6, "M tok/s")
    print("  Token count:", token_count)

    print("\n--- vs HuggingFace transformers ---")
    print("  HuggingFace: 0.98 M tok/s")
    print("  mojo-tokenizer:", tok_per_sec / 1e6, "M tok/s")
    print("  Speedup:", tok_per_sec / 0.98e6, "x")


fn test_mistral() raises:
    """Test Mistral 7B tokenizer."""
    print("\n" + "=" * 70)
    print("Testing: Mistral 7B (mistralai/Mistral-7B-v0.1)")
    print("=" * 70)

    var tokenizer_path = "data/huggingface/mistralai_Mistral-7B-v0.1/tokenizer.json"

    print("  Loading tokenizer.json...")
    var start = perf_counter_ns()
    var tokenizer = BPETokenizer.from_huggingface(tokenizer_path)
    var load_time = Float64(perf_counter_ns() - start) / 1e6

    print("  Vocab size:", tokenizer.vocab_size())
    print("  Load time:", load_time, "ms")

    print("\n--- Correctness Test ---")

    # Test "Hello, world!" -> [22557, 28725, 1526, 28808]
    var text1 = "Hello, world!"
    var tokens1 = tokenizer.encode(text1)
    var expected1 = List[Int](22557, 28725, 1526, 28808)
    var matches1 = tokens_match(tokens1, expected1)

    print("  'Hello, world!' ->", "✓" if matches1 else "✗")
    if not matches1:
        print("    Expected:", end="")
        print_tokens(expected1)
        print("    Actual:  ", end="")
        print_tokens(tokens1)

    print("\n  Passed:", 1 if matches1 else 0, "/ 1")

    # Performance benchmark
    print("\n--- Performance Benchmark ---")
    var sherlock = read_file("data/sherlock.txt")
    print("  Using: sherlock.txt (", len(sherlock), "chars)")

    var result = benchmark_encode(tokenizer, sherlock, iterations=20, warmup=3)
    var tok_per_sec = result[0]
    var token_count = result[1]

    print("  Encoding:", tok_per_sec / 1e6, "M tok/s")
    print("  Token count:", token_count)

    print("\n--- vs HuggingFace transformers ---")
    print("  HuggingFace: 1.06 M tok/s")
    print("  mojo-tokenizer:", tok_per_sec / 1e6, "M tok/s")
    print("  Speedup:", tok_per_sec / 1.06e6, "x")


fn main() raises:
    print("=" * 70)
    print("HuggingFace BPE Validation - mojo-tokenizer")
    print("=" * 70)
    print()

    test_qwen()
    test_mistral()

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
