# Architecture

The package has four focused areas. Dependencies point inward toward core data
structures; ONNX and downstream model runtimes do not leak into tokenization.

```text
src/
  core/             BPE vocabulary, encoding, diagnostic decoding, internals
  preprocessing/    internal byte-boundary helpers
  training/         deterministic byte-level BPE vocabulary training
  onnx_utils/       fixed-shape tensor staging for ONNX integration
```

## Core

`BpeTokenizer` is the public encoding facade. It recognizes registered special
tokens, then optional deterministic byte-range patterns, and delegates
remaining text to `BpeEncoder`. `encode()` retains the `List[Int]` contract;
`encode_with_spans()` returns token IDs aligned to half-open UTF-8 byte offsets.

`BpeEncoder` owns the BPE algorithm and its state. `Vocabulary` stores token
mappings and merge metadata. Cache, trie, bitfield, and backtracking modules are
internal implementation details rather than public workflow concepts. Frozen
byte-trie nodes refer to contiguous edge intervals. Exact bytes use singleton
intervals; wide prefixes are promoted to per-prefix `IntervalTree` instances
owned centrally by the trie.

Its byte-to-Unicode table and merge working sets use `List` because they hold
temporary or nonnumeric tokenizer state. The token cache stores relative
`TokenSpan` results, while reusable backtracking slots retain allocation-backed
token IDs and spans. Public callers can request `List[Int]` or
`List[TokenSpan]`.

`BytePatternSet` registers fixed-length sequences of non-overlapping byte
intervals. Pattern IDs must already exist in `Vocabulary`; their canonical text
is diagnostic output, so ranged pattern encoding is intentionally not
source-reversible. Patterns are matched only on UTF-8 code-point boundaries.

`TokenDecoder` is separate from the encoder because inference capture is
encoding-focused. It remains available for diagnostics and round-trip tests.

## Internal boundary helpers

The BPE encoder uses the `preprocessing/whitespace.mojo` helpers to construct
byte-boundary masks. Normalization, pre-tokenization, and model-specific
post-processing are not exposed because they were disconnected prototypes and
are not part of the current encoding pipeline.

## Training

`BpeTrainer` constructs deterministic byte-level vocabularies from
caller-provided strings. It stays separate from inference while sharing the
core `Vocabulary` type.

## ONNX utilities

`OnnxTens` wraps caller-owned scalar spans as fixed batch and row views. Its
batch path copies disjoint row ranges into a contiguous allocation in parallel;
its vector path copies one row for low-latency request handling.

This boundary currently stages typed tensor values only. ONNX `TensorProto`
encoding, model construction, padding, masks, and model execution policy remain
separate planned work.

## Public API

The root package exports the encoding facade, vocabulary, special-token,
byte-pattern, and token-span types, diagnostic decoder, ONNX tensor utilities,
and deterministic BPE trainer. Vocabularies are still supplied
programmatically; tokenizer-file loaders and compatibility constructors are
not implemented yet.
