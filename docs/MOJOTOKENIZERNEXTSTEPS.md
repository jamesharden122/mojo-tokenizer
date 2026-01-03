# Mojo Tokenizer Next Steps

## Executive Summary

**Status**: Package compiles on Mojo 25.4.0, but **BPE merges are not being applied correctly**.

**Critical Bug**: The tokenizer falls back to byte-level encoding instead of proper BPE merges.

| Metric | Expected | Actual | Status |
|--------|----------|--------|--------|
| "Hello, world!" tokens | 4 (Python HuggingFace) | 13+ (byte fallback) | **BROKEN** |
| HuggingFace loading | Works | Works | OK |
| Vocab size (Qwen3) | 151,643 | 151,643 | OK |
| Load time | <100ms | ~341ms | Needs work |

## Root Cause Analysis

### The Core Problem

The BPE encoding algorithm (`_bpe_encode` in `src/bpe.mojo:375-459`) converts input text through a GPT-2 style byte encoder before looking up merges. However, **HuggingFace vocabularies store tokens directly without this encoding**.

```
Input: "Hello"
Expected flow:
  1. Vocab lookup: "Hello" → ID 9707 (direct match)

Actual flow (broken):
  1. Byte encoding: [72, 101, 108, 108, 111] → ["H", "e", "l", "l", "o"] (GPT-2 unicode)
  2. Merge lookup: find("H" + "e") → -1 (not in vocab's merge format)
  3. Fallback: return individual byte IDs → [39, 68, 75, 75, 78] (13+ tokens)
```

### Why Python Works

Python's HuggingFace tokenizers detect the vocabulary format and skip byte-level encoding when:
1. The vocab contains direct word entries (like "Hello", "world")
2. The pre-tokenizer is NOT ByteLevel
3. The model type indicates word-level BPE (not byte-level BPE)

Our Mojo implementation **always** applies byte-level encoding, which is incorrect for most HuggingFace models.

### Evidence from Debug Sessions

```
# Python (correct)
>>> tokenizer.encode("Hello, world!")
[9707, 11, 1879, 0]  # 4 tokens: "Hello", ",", " world", "!"

# Mojo (broken)
>>> tokenizer.encode("Hello, world!")
[39, 68, 75, 75, 78, 11, 220, 86, 78, 81, 75, 67, 0]  # 13+ byte tokens
```

---

## Phase 0: Critical Bug Fix (MUST DO FIRST)

**Goal**: Achieve correct tokenization before any performance optimization.

### Task 0.1: Add Format Detection

Detect whether the loaded vocabulary uses byte-level or word-level BPE:

```mojo
# src/formats/huggingface.mojo
struct HuggingFaceConfig:
    var is_byte_level: Bool
    var add_prefix_space: Bool
    var pretokenizer_type: String  # "ByteLevel", "Whitespace", etc.
```

Parse the `pre_tokenizer` section from tokenizer.json:
```json
{
  "pre_tokenizer": {
    "type": "ByteLevel",
    "add_prefix_space": false,
    "trim_offsets": true
  }
}
```

### Task 0.2: Implement Word-Level BPE Path

When vocabulary is NOT byte-level:
1. Skip the `_byte_encoder` transformation entirely
2. Look up tokens directly in vocabulary
3. Apply merges using the vocab's native format

```mojo
fn _bpe_encode(mut self, word: String) raises -> List[Int]:
    if not self._is_byte_level:
        return self._word_level_encode(word)
    else:
        return self._byte_level_encode(word)  # Current implementation
```

### Task 0.3: Handle GPT-2 Ġ Prefix

Many HuggingFace vocabularies use `Ġ` (U+0120) to mark space prefixes:
- "world" in vocab might be stored as "Ġworld" (space-prefixed)
- Need to handle this in pre-tokenization

```mojo
fn _add_gpt2_prefix(self, word: String, is_first: Bool) -> String:
    if not is_first and self._add_prefix_space:
        return "Ġ" + word
    return word
```

### Task 0.4: Validation Suite

Create comprehensive tests comparing Mojo vs Python outputs:

```mojo
# tests/test_huggingface_parity.mojo
fn test_qwen3_encoding():
    var mojo_tokens = tokenizer.encode("Hello, world!")
    var expected = List[Int](9707, 11, 1879, 0)  # From Python
    assert_equal(mojo_tokens, expected)
```

**Acceptance Criteria**: 100% match with Python HuggingFace on test corpus.

---

## Phase 1: Performance Investigation (After Bug Fix)

Following mojo-json's investigation methodology.

### 1.1 Establish Baseline Metrics

```bash
# Real-world benchmark (not synthetic)
mojo run benchmarks/huggingface_benchmark.mojo

# Metrics to capture:
- Unique texts (no cache): X chars/sec
- Repeated texts (warm cache): Y chars/sec
- Memory usage during encoding
- Load time for various vocab sizes
```

### 1.2 Profile Hot Paths

Key areas from mojo-json learnings:

| Area | Likely Issue | Investigation |
|------|--------------|---------------|
| String concatenation | O(n²) in merge loop | Profile `tokens[i] + tokens[i + 1]` |
| Dict lookups | Hash collisions | Profile `vocab.get_id()` |
| List operations | Resize overhead | Check pre-allocation effectiveness |
| Byte encoding | Per-character allocation | Profile `_byte_encoder[byte_val]` |

### 1.3 Identify O(n²) Patterns

From mojo-json's findings, watch for:
- Repeated string concatenation in loops
- List append without pre-allocation
- Dict rebuilding during iteration
- Copy-on-access semantics

---

## Phase 2: Quick Wins (Low Risk, High Impact)

Based on mojo-json's successful optimizations.

### 2.1 Pre-sized Collections

```mojo
# Before (current)
var result = List[Int]()

# After
var result = List[Int](capacity=len(text) // 4)  # ~4 chars per token average
```

### 2.2 Merge Loop Buffer Reuse

Current implementation already has buffer reuse (`src/bpe.mojo:404-444`), but verify:
- Buffer capacity is maintained across iterations
- No intermediate allocations in `tokens[i] + tokens[i + 1]`

### 2.3 String Builder for Concatenation

Replace:
```mojo
unicode_text += parts[i]  # O(n²)
```

With:
```mojo
var builder = StringBuilder(capacity=len(parts) * 8)
for i in range(len(parts)):
    builder.write(parts[i])
return builder.to_string()
```

### 2.4 Batch Allocation for Byte Encoding

```mojo
# Current: allocate per byte
for i in range(len(byte_text)):
    unicode_parts.append(self._byte_encoder[byte_val])

# Optimized: single allocation
var total_len = 0
for i in range(len(byte_text)):
    total_len += len(self._byte_encoder[byte_text[i]])
var result = String(capacity=total_len)
```

---

## Phase 3: Vocabulary Architecture (Medium Risk)

### 3.1 Current Architecture Analysis

```
src/vocab.mojo:
  - Dict[String, Int] _text_to_id    # Token lookup
  - Dict[Int, String] _id_to_text    # Reverse lookup
  - Dict[String, Int] _merges        # Merge rank lookup
```

**Issues**:
- Three separate Dict lookups for common operations
- String keys cause hash overhead
- No locality of reference

### 3.2 Tape-Based Vocabulary (from mojo-json)

Following simdjson's tape architecture:

```mojo
struct TapeVocabulary:
    var tape: List[UInt8]           # Packed token strings
    var offsets: List[Int]          # Start position of each token
    var lengths: List[UInt16]       # Length of each token
    var merge_ranks: List[Int]      # Merge rank by token ID
    var hash_table: List[Int]       # Hash → token ID
```

**Benefits**:
- Single contiguous allocation
- Cache-friendly sequential access
- O(1) ID→text lookup (just offset + length)
- Reduced memory fragmentation

### 3.3 Perfect Hashing for Vocabulary

Since vocabulary is fixed after loading:

```mojo
struct PerfectHashVocab:
    """Minimal perfect hash for O(1) guaranteed lookups."""
    var data: List[UInt8]           # Token strings
    var hash_params: List[Int]      # MPH parameters

    fn get_id(self, token: String) -> Int:
        var h = self._hash(token)
        return h if self._verify(h, token) else -1
```

---

## Phase 4: SIMD Optimizations (from mojo-json)

### 4.1 SIMD Word Boundary Detection

Current implementation (`src/bpe.mojo:318-353`) is scalar. Apply SIMD:

```mojo
fn _split_into_words_simd(self, text: String) -> List[String]:
    alias WIDTH = 32
    var words = List[String]()
    var ptr = text.unsafe_ptr()
    var n = len(text)

    var i = 0
    while i + WIDTH <= n:
        var chunk = ptr.load[width=WIDTH](i)

        # SIMD comparison for boundaries
        var is_space = chunk == SIMD[DType.uint8, WIDTH](32)
        var is_punct = (chunk >= 33) & (chunk <= 47)
        var boundaries = is_space | is_punct

        # Extract boundary positions
        var mask = boundaries.cast[DType.uint32]().reduce_or()
        while mask != 0:
            var pos = count_trailing_zeros(mask)
            # ... extract word
            mask &= mask - 1

        i += WIDTH

    # Scalar fallback for remainder
    ...
```

### 4.2 SIMD Hash Computation

For vocabulary lookups:

```mojo
fn hash_simd(text: String) -> UInt64:
    alias WIDTH = 8
    var ptr = text.unsafe_ptr()
    var h = SIMD[DType.uint64, WIDTH](0)
    var mul = SIMD[DType.uint64, WIDTH](0x517cc1b727220a95)

    var i = 0
    while i + WIDTH <= len(text):
        var chunk = ptr.load[width=WIDTH](i).cast[DType.uint64]()
        h = (h ^ chunk) * mul
        i += WIDTH

    return h.reduce_xor()
```

### 4.3 SIMD Merge Candidate Detection

Find mergeable pairs in parallel:

```mojo
fn find_best_merge_simd(self, tokens: List[Int]) -> Int:
    """Find highest-priority merge using SIMD."""
    alias WIDTH = 8

    var best_idx = -1
    var best_rank = Int.MAX

    var i = 0
    while i + WIDTH < len(tokens):
        # Load WIDTH consecutive pairs
        var first_ids = ...  # tokens[i:i+WIDTH]
        var second_ids = ... # tokens[i+1:i+WIDTH+1]

        # Parallel merge rank lookup
        var ranks = self._lookup_merge_ranks_simd(first_ids, second_ids)

        # Find minimum valid rank
        var valid = ranks >= 0
        var min_rank = (ranks | ~valid).reduce_min()
        if min_rank < best_rank:
            best_rank = min_rank
            best_idx = i + find_first_set(ranks == min_rank)

        i += WIDTH

    return best_idx
```

---

## Phase 5: Cache Optimization

### 5.1 Current Cache Analysis

```
src/cache/token_cache.mojo:
  - LRU cache with Dict[String, List[Int]]
  - 10k entry default
  - String keys (allocation overhead)
```

### 5.2 Cache Key Optimization

Replace string keys with hashes:

```mojo
struct OptimizedTokenCache:
    var entries: List[CacheEntry]
    var hash_to_idx: Dict[UInt64, Int]

    fn get(self, word: String) -> Optional[List[Int]]:
        var h = hash_simd(word)
        if h in self.hash_to_idx:
            var idx = self.hash_to_idx[h]
            if self.entries[idx].word == word:  # Verify (collision check)
                return self.entries[idx].tokens
        return None
```

### 5.3 Inline Small Results

For common short words:

```mojo
struct InlineCacheEntry:
    var word_hash: UInt64
    var token_count: UInt8
    var inline_tokens: SIMD[DType.int32, 8]  # Up to 8 tokens inline
    var overflow: Optional[List[Int]]        # For longer results
```

---

## Phase 6: GPU Acceleration (Future)

From mojo-json's GPU pipeline design.

### 6.1 Batch Tokenization on GPU

```mojo
struct GPUTokenizer:
    var vocab_gpu: GPUBuffer[UInt8]      # Vocabulary on GPU
    var hash_table_gpu: GPUBuffer[Int]   # Hash table on GPU

    fn encode_batch_gpu(
        self,
        texts: List[String],
        stream: GPUStream
    ) -> GPUBuffer[Int]:
        """Parallel tokenization of entire batch."""
        # 1. Copy texts to GPU
        # 2. Parallel pre-tokenization (word splitting)
        # 3. Parallel vocabulary lookup
        # 4. Parallel BPE merge (tricky - data-dependent)
        # 5. Return results
```

### 6.2 GPU-Friendly BPE

The BPE merge loop is inherently sequential (data-dependent), but can be parallelized across batch:

```mojo
# Parallel across batch (one GPU thread per text)
@gpu
fn bpe_encode_kernel(
    texts: GPUBuffer[UInt8],
    text_offsets: GPUBuffer[Int],
    vocab: GPUBuffer[UInt8],
    results: GPUBuffer[Int],
):
    var text_idx = gpu.thread_id.x
    if text_idx >= len(text_offsets) - 1:
        return

    var start = text_offsets[text_idx]
    var end = text_offsets[text_idx + 1]

    # BPE encode this text (sequential within thread)
    ...
```

---

## Performance Targets

| Metric | Current | Phase 2 Target | Phase 4 Target | Phase 6 Target |
|--------|---------|----------------|----------------|----------------|
| Throughput (chars/sec) | N/A (broken) | 500k | 2M | 10M (GPU) |
| Memory (vocab load) | ~50MB | ~30MB | ~20MB | ~20MB |
| Load time | ~341ms | <200ms | <100ms | <100ms |
| Cache hit rate | ~80% | ~85% | ~90% | N/A |

---

## Implementation Priority

### Immediate (Week 1)
1. **Phase 0**: Fix BPE correctness bug - nothing else matters until this works
2. Create parity test suite against Python HuggingFace

### Short-term (Week 2-3)
3. **Phase 1**: Profile and establish baseline
4. **Phase 2**: Quick wins (pre-allocation, string builder)

### Medium-term (Month 1-2)
5. **Phase 3**: Tape-based vocabulary architecture
6. **Phase 4**: SIMD optimizations

### Long-term (Month 3+)
7. **Phase 5**: Advanced cache optimizations
8. **Phase 6**: GPU acceleration

---

## Lessons from mojo-json

Key patterns that transfer directly:

1. **Two-Stage Architecture**: Parse structure first (vocabulary), then access on-demand
2. **Tape Data Structure**: Pack related data contiguously for cache efficiency
3. **SIMD Everything**: Character classification, hashing, comparison
4. **Lazy Initialization**: Don't build indexes until needed
5. **Buffer Reuse**: Pre-allocate and clear instead of allocate per operation
6. **Reference Semantics**: Avoid copies in hot paths
7. **Profile Before Optimizing**: Measure, don't guess

---

## Files to Create/Modify

### Phase 0 (Bug Fix)
- `src/formats/huggingface.mojo` - Add format detection
- `src/bpe.mojo` - Add word-level BPE path
- `tests/test_huggingface_parity.mojo` - Validation suite

### Phase 2 (Quick Wins)
- `src/bpe.mojo` - String builder, pre-allocation
- `src/cache/token_cache.mojo` - Cache key optimization

### Phase 3 (Architecture)
- `src/vocab_tape.mojo` - New tape-based vocabulary
- `src/hash/perfect_hash.mojo` - MPH implementation

### Phase 4 (SIMD)
- `src/simd/word_split.mojo` - SIMD word boundary detection
- `src/simd/hash.mojo` - SIMD hashing
- `src/simd/merge.mojo` - SIMD merge detection

### Phase 6 (GPU)
- `src/gpu/tokenizer.mojo` - GPU tokenization kernels
- `src/gpu/vocab.mojo` - GPU vocabulary management

---

## References

- [mojo-json PARSER_PERFORMANCE_INVESTIGATION.md](../../mojo-json/docs/PARSER_PERFORMANCE_INVESTIGATION.md)
- [mojo-json ANALYSIS.md](../../mojo-json/docs/ANALYSIS.md)
- [simdjson Paper](https://arxiv.org/abs/1902.08318) - Tape architecture
- [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers) - Reference implementation
- [tiktoken](https://github.com/openai/tiktoken) - OpenAI's byte-level BPE
