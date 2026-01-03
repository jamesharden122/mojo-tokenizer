"""
Benchmark for BPE tokenizer encoding performance.

Target: 100k+ tokens/sec on M3 Ultra
"""

from time import perf_counter_ns


fn benchmark_encode() raises:
    """Benchmark encoding throughput."""
    print("=== BPE Encode Benchmark ===\n")

    # Test texts of varying sizes
    var medium_text = "The quick brown fox jumps over the lazy dog. " * 10

    print("Test 1: Word splitting performance")
    print("-" * 40)

    var iterations = 10000

    # Benchmark current approach (char-by-char)
    var start = perf_counter_ns()
    for _ in range(iterations):
        var words = split_words_slow(medium_text)
        _ = words
    var slow_time = perf_counter_ns() - start

    # Benchmark optimized approach (slice-based)
    start = perf_counter_ns()
    for _ in range(iterations):
        var words = split_words_fast(medium_text)
        _ = words
    var fast_time = perf_counter_ns() - start

    print("Slow (char-by-char):", slow_time // 1000, "µs total")
    print("Fast (slice-based): ", fast_time // 1000, "µs total")
    if fast_time > 0:
        print("Speedup:", Float64(slow_time) / Float64(fast_time), "x")
    print()

    print("Test 2: Byte encoding lookup")
    print("-" * 40)

    # Benchmark Dict lookup
    var byte_dict = Dict[Int, String]()
    for i in range(256):
        if i >= 32 and i < 127:
            byte_dict[i] = chr(i)
        else:
            byte_dict[i] = chr(256 + i)

    start = perf_counter_ns()
    for _ in range(iterations):
        for i in range(256):
            try:
                var _ = byte_dict[i]
            except:
                pass
    var dict_time = perf_counter_ns() - start

    # Benchmark List lookup (array-style)
    var byte_list = List[String](capacity=256)
    for i in range(256):
        if i >= 32 and i < 127:
            byte_list.append(chr(i))
        else:
            byte_list.append(chr(256 + i))

    start = perf_counter_ns()
    for _ in range(iterations):
        for i in range(256):
            var _ = byte_list[i]
    var list_time = perf_counter_ns() - start

    print("Dict lookup:", dict_time // 1000, "µs total")
    print("List lookup:", list_time // 1000, "µs total")
    if list_time > 0:
        print("Speedup:", Float64(dict_time) / Float64(list_time), "x")
    print()

    print("Test 3: Merge operations")
    print("-" * 40)

    # Benchmark new list creation per merge
    start = perf_counter_ns()
    for _ in range(iterations):
        var tokens = List[String]()
        tokens.append("a")
        tokens.append("b")
        tokens.append("c")
        tokens.append("d")

        # Simulate 3 merges (create new list each time)
        for merge_idx in range(3):
            var new_tokens = List[String]()
            var i = 0
            while i < len(tokens):
                if i == merge_idx and i + 1 < len(tokens):
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens^
    var new_list_time = perf_counter_ns() - start

    # Benchmark in-place merge (pre-allocate)
    start = perf_counter_ns()
    for _ in range(iterations):
        var tokens = List[String](capacity=10)
        tokens.append("a")
        tokens.append("b")
        tokens.append("c")
        tokens.append("d")

        # Simulate 3 merges (reuse buffer)
        var buffer = List[String](capacity=10)
        for merge_idx in range(3):
            buffer.clear()
            var i = 0
            while i < len(tokens):
                if i == merge_idx and i + 1 < len(tokens):
                    buffer.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    buffer.append(tokens[i])
                    i += 1
            # Swap instead of reassign
            var temp = tokens^
            tokens = buffer^
            buffer = temp^
    var reuse_time = perf_counter_ns() - start

    print("New list per merge:", new_list_time // 1000, "µs total")
    print("Reuse buffer:      ", reuse_time // 1000, "µs total")
    if reuse_time > 0:
        print("Speedup:", Float64(new_list_time) / Float64(reuse_time), "x")
    print()


fn split_words_slow(text: String) -> List[String]:
    """Original slow implementation - char-by-char concat."""
    var words = List[String]()
    var current_word = String()

    for i in range(len(text)):
        var c = text[i]
        var code = ord(c)

        var is_boundary = (code == 32 or
                          (code >= 33 and code <= 47) or
                          (code >= 58 and code <= 64) or
                          (code >= 91 and code <= 96) or
                          (code >= 123 and code <= 126))

        if is_boundary:
            if len(current_word) > 0:
                words.append(current_word)
                current_word = String()
            words.append(String(c))
        else:
            current_word += c  # O(n²) !

    if len(current_word) > 0:
        words.append(current_word)

    return words^


fn split_words_fast(text: String) -> List[String]:
    """Optimized implementation - slice-based."""
    var words = List[String]()
    var start = 0
    var n = len(text)

    for i in range(n):
        var code = ord(text[i])

        var is_boundary = (code == 32 or
                          (code >= 33 and code <= 47) or
                          (code >= 58 and code <= 64) or
                          (code >= 91 and code <= 96) or
                          (code >= 123 and code <= 126))

        if is_boundary:
            if i > start:
                words.append(String(text[start:i]))  # Single slice allocation
            words.append(String(text[i]))  # Single char
            start = i + 1

    if start < n:
        words.append(String(text[start:n]))

    return words^


fn main() raises:
    benchmark_encode()
