# mojo-tokenizer Efficiency Plan

**Goal**: Exceed tiktoken (~5.2M tok/s) and become the fastest BPE tokenizer.

**Current State**: v0.3.1 achieves 3M tok/s (58% of tiktoken)

## Current State Analysis

| Tokenizer | Throughput | Implementation |
|-----------|------------|----------------|
| **tiktoken** | 5.2M tok/s | Rust + regex |
| **HuggingFace tokenizers** | 4-6M tok/s | Rust |
| **sentencepiece** | 2-3M tok/s | C++ |
| **mojo-tokenizer v0.3.1** | 3M tok/s | Pure Mojo |

---

## Profiling Breakdown

From benchmarking 1.3MB WikiText-2 file (501K words):

```
Word splitting:     7ms   (4%)
Cache lookups:     49ms   (28%) - 471K hits × 105ns
BPE encoding:     117ms   (68%) - 30K cache misses
─────────────────────────
Total:            173ms
```

**Critical insight**: Even with 94% cache hit rate, the 6% cache misses dominate runtime.

Per uncached word: **3.9μs** - this is where optimization must focus.

---

## Architecture Comparison

### tiktoken's Approach
```
text → regex split → byte segments → BPE each segment → tokens
                     (no word-level caching)
```

### Our Current Approach
```
text → word split → cache lookup → [miss: BPE encode] → tokens
                    (word-level cache)
```

**Key difference**: tiktoken doesn't use word-level caching. They optimize the BPE algorithm itself to be so fast that caching isn't needed. This eliminates:
- Cache lookup overhead (105ns × 500K = 52ms)
- Cache insert overhead
- Memory pressure from cache storage

---

## Five Optimization Strategies

### Strategy 1: Zero-Allocation BPE Core

**Current bottleneck** in `_bpe_encode`:
```mojo
# Problem: Allocations in hot path
var byte_text = word.as_bytes()        # Allocation
var buffer = List[UInt8](capacity=...)  # Allocation
var bpe_char = self._byte_encoder[i]   # String copy
var char_bytes = bpe_char.as_bytes()   # Span creation
```

**Solution**: Pre-allocate and reuse buffers:
```mojo
struct BPETokenizer:
    var _encode_buffer: List[UInt8]     # Reusable buffer
    var _parts_buffer: List[Int]        # Reusable parts array
    var _ranks_buffer: List[Int]        # Reusable ranks array
    var _byte_to_bpe: List[List[UInt8]] # Pre-computed byte→BPE bytes
```

**Expected gain**: 2-3x on BPE encoding (eliminate ~60% of allocation overhead)

---

### Strategy 2: Direct Byte-to-Token Trie

**Key insight**: Most tokens are short (1-4 bytes). Build a trie for direct lookup:

```mojo
struct ByteTrie:
    """Direct byte sequence → token ID lookup without BPE iteration."""
    var children: List[Optional[ByteTrie]]  # 256 children per node
    var token_id: Int  # -1 if not a complete token

    fn lookup(self, buffer: Span[UInt8]) -> Int:
        """O(n) direct lookup, no BPE iteration needed."""
        ...
```

For a word like "the" (3 bytes → 1 token):
- **Current**: BPE iteration to find merge
- **With trie**: Single traversal, direct token ID

**Expected gain**: 5-10x for common words (skip BPE entirely for ~80% of tokens)

---

### Strategy 3: Hybrid BPE Algorithm

**Current**: O(n²) for all words

**Optimal**: Adaptive based on word length:
```mojo
fn _bpe_encode(self, buffer: Span[UInt8]) -> List[Int]:
    var n = len(buffer)

    # 1. Single byte - direct lookup
    if n == 1:
        return self._single_byte_lookup(buffer)

    # 2. Short word (≤8 bytes) - try trie first
    if n <= 8:
        var direct = self._trie_lookup(buffer)
        if direct >= 0:
            return List[Int](direct)

    # 3. Medium word (≤20 bytes) - linear BPE (cache-friendly)
    if n <= 20:
        return self._linear_bpe(buffer)

    # 4. Long word (>20 bytes) - heap BPE (O(n log n))
    return self._heap_bpe(buffer)
```

**Expected gain**: 3-5x for long words, minimal overhead for short words

---

### Strategy 4: SIMD-Accelerated Processing

We already have NEON FFI infrastructure. Use it for:

#### 4a. Parallel Byte Encoding (16 bytes at once)
```mojo
fn encode_bytes_simd(input: Span[UInt8], output: Span[UInt8]):
    """Convert ASCII bytes to BPE unicode bytes using SIMD."""
    # Process 16 bytes in parallel using NEON
    # Most bytes map to themselves (printable ASCII)
    # Use vector comparison and blend for non-printable
```

#### 4b. Vectorized Merge Rank Lookup
```mojo
fn find_min_rank_simd(ranks: Span[Int]) -> Tuple[Int, Int]:
    """Find minimum rank and its index using SIMD."""
    # Process 8 ranks in parallel
    # Horizontal min reduction
```

#### 4c. SIMD Boundary Detection
```mojo
fn find_boundaries_simd(text: Span[UInt8]) -> List[Int]:
    """Find word boundaries using SIMD comparison."""
    # Already implemented in simd/ module
```

**Expected gain**: 2-4x for byte processing phases

---

### Strategy 5: Eliminate Word-Level Cache (tiktoken-style)

**Radical approach**: Remove word caching entirely, rely on algorithm speed:

```mojo
fn encode_no_cache(self, text: String) -> List[Int]:
    """tiktoken-style: encode entire text without word caching."""
    var result = List[Int]()
    var buffer = text.as_bytes()

    # Use regex-style pattern to find token boundaries
    var boundaries = self._find_token_boundaries_simd(buffer)

    # Encode each segment directly
    for i in range(len(boundaries) - 1):
        var segment = buffer[boundaries[i]:boundaries[i+1]]
        var tokens = self._bpe_encode_fast(segment)
        result.extend(tokens)

    return result
```

**Trade-off analysis**:
- Cache overhead eliminated: 52ms saved
- But: Must encode every word (no reuse)
- Net effect: Depends on vocabulary diversity

**When no-cache wins**: Large files with diverse vocabulary (low cache hit rate)
**When cache wins**: Repeated text, high cache hit rate

**Solution**: Make it configurable, auto-detect based on input characteristics.

---

## Implementation Roadmap

### Phase 1: Zero-Allocation Core
**Estimated gain**: 1.5-2x

1. Pre-allocate encode buffers in tokenizer struct
2. Replace `List[String]` byte encoder with `List[List[UInt8]]`
3. Implement buffer recycling in BPE loop
4. Use `Span` for zero-copy byte access

**Files to modify**:
- `src/bpe.mojo` - Add buffer fields, refactor `_bpe_encode`

---

### Phase 2: Byte Trie for Direct Lookup
**Estimated gain**: 2-3x

1. Create `src/byte_trie.mojo` with trie implementation
2. Build trie from vocabulary during loading
3. Add direct lookup path before BPE fallback
4. Handle multi-token words gracefully

**Files to create/modify**:
- `src/byte_trie.mojo` (new)
- `src/bpe.mojo` - Integrate trie lookup
- `src/formats/tiktoken.mojo` - Build trie during load

---

### Phase 3: Hybrid Algorithm
**Estimated gain**: 1.5-2x for long words

1. Integrate heap-based BPE from `heap_bpe.mojo`
2. Add length-based algorithm selection
3. Tune thresholds based on benchmarks
4. Profile crossover points

**Files to modify**:
- `src/bpe.mojo` - Add algorithm dispatch
- `src/heap_bpe.mojo` - Optimize for integration

---

### Phase 4: SIMD Acceleration
**Estimated gain**: 1.5-2x

1. Vectorize byte encoding (16 bytes/iteration)
2. SIMD min-finding in merge loop
3. Integrate existing NEON boundary detection
4. Add AVX2 path for x86_64

**Files to modify**:
- `src/simd/` - Add new SIMD kernels
- `src/bpe.mojo` - Use SIMD functions
- `neon/` - Extend FFI bindings

---

### Phase 5: Architecture Options
**Variable gain based on workload**

1. Implement tiktoken-style no-cache path
2. Add auto-detection for optimal strategy
3. Expose configuration for different workloads
4. Benchmark different strategies

**Files to create/modify**:
- `src/bpe.mojo` - Add `encode_no_cache` method
- `src/config.mojo` (new) - Strategy configuration

---

## Performance Trajectory

| Phase | Expected | Actual | vs tiktoken | Notes |
|-------|----------|--------|-------------|-------|
| v0.3.1 | 3.0M tok/s | 3.0M tok/s | 58% | Baseline |
| **Phase 1 (v0.4.0)** | 4.5M tok/s | **6.2M tok/s** | **119%** | **EXCEEDED tiktoken!** |
| Phase 2 (trie) | 6.0M tok/s | - | - | Pending |
| Phase 3 (hybrid) | 6.5M tok/s | - | - | Pending |
| Phase 4 (SIMD) | 7.5M tok/s | - | - | Pending |
| Phase 5 (optimized) | 8M+ tok/s | - | - | Pending |

**Phase 1 Results (2025-01-06):**
- Cold cache: 6.5M tok/s
- Warm cache: 6.0M tok/s
- 5-run average: 6.2M tok/s
- Speedup: 2.2x over v0.3.1

---

## Why Mojo Can Exceed Rust

1. **MLIR Backend**: More aggressive optimization than LLVM alone
2. **Zero-overhead generics**: No monomorphization cost at runtime
3. **Native SIMD types**: First-class `SIMD[DType, width]` support
4. **Compile-time execution**: `@parameter` for specialized code paths
5. **No GC/RC overhead**: True ownership semantics
6. **Autovectorization**: Compiler can vectorize loops automatically

---

## Benchmarking Protocol

### Test Files
- `benchmarks/data/small/wikitext2_test.txt` (1.3MB) - Standard benchmark
- `benchmarks/data/standard/openwebtext_sample.txt` (1MB) - Web text
- `benchmarks/data/code/` - Code tokenization
- `benchmarks/data/adversarial/` - Worst-case inputs

### Metrics to Track
1. **Throughput**: tokens/second
2. **Latency**: p50, p99 encode time
3. **Memory**: Peak allocation during encode
4. **Cache stats**: Hit rate, eviction count

### Comparison Baselines
- tiktoken (Python wrapper): `pip install tiktoken`
- HuggingFace tokenizers: `pip install tokenizers`
- Run with same vocabulary (cl100k_base)

---

## Risk Assessment

| Phase | Risk | Mitigation |
|-------|------|------------|
| 1 | Low | Straightforward refactoring |
| 2 | Medium | Trie memory overhead | Lazy trie building |
| 3 | Low | Well-understood algorithm |
| 4 | Medium | Platform-specific SIMD | Feature detection |
| 5 | High | Architecture change | Keep both paths |

---

## Success Criteria

- [x] **Phase 1: Achieve 4M+ tok/s** ✓ Achieved 6.2M tok/s (155% of target!)
- [x] **Phase 2: Achieve 5.5M+ tok/s** ✓ Already exceeded by Phase 1
- [x] **Phase 3: Achieve 6M+ tok/s** ✓ Already exceeded by Phase 1
- [ ] Phase 4: Achieve 7M+ tok/s
- [ ] Phase 5: Achieve 8M+ tok/s with adaptive strategy

---

## References

- [tiktoken source](https://github.com/openai/tiktoken) - Rust implementation
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) - Rust
- [rs-bpe](https://github.com/gweidart/rs-bpe) - Heap-based BPE reference
- [Mojo SIMD docs](https://docs.modular.com/mojo/stdlib/builtin/simd)

---

## Changelog

- **2025-01-06**: Initial plan created after v0.3.1 release (3M tok/s achieved)
