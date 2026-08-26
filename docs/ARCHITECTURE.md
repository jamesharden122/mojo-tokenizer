# Architecture

The package has three explicit layers. Dependencies point inward toward core
data structures; ONNX and SurrealML do not leak into tokenization.

```text
src/
  core/             BPE vocabulary, encoding, diagnostic decoding, internals
  preprocessing/    text and token-sequence transformations
  model_input/      future typed model-input contracts
  io/               external data-source integrations
```

## Core

`BpeTokenizer` is the public encoding facade. It recognizes registered special
tokens and delegates ordinary text to `BpeEncoder`.

`BpeEncoder` owns the BPE algorithm and its state. `Vocabulary` stores token
mappings and merge metadata. Cache, trie, bitfield, and backtracking modules are
internal implementation details rather than public workflow concepts.

`TokenDecoder` is separate from the encoder because model-input construction is
encoding-focused. It remains available for diagnostics and round-trip tests.

## Preprocessing

Normalizers transform source text. Pre-tokenizers divide source text into
model-defined pieces. Post-processors transform token sequences and add
model-defined special-token arrangements. SIMD helpers live beside the stages
that consume them.

The current BPE facade retains its existing boundary splitting internally. A
future behavior change may inject preprocessing stages explicitly, but this
refactor does not alter that algorithm.

## Model input

`ModelInputConfig`, `NamedInt64Tensor`, and `TokenizedInputs` define the future
boundary between tokenization and model execution. Tensor values are owned by
allocator-backed `Allocation[Int64]` buffers; `List` remains appropriate for
small shape metadata and temporary token collections. `ModelInputBuilder`
declares the construction interface.

Padding, truncation, attention-mask generation, ONNX execution, and SurrealML
serialization are deliberately not implemented yet.

## I/O

`io` is the boundary for external tokenizer data sources. It currently exposes
the sibling `read_bin` package's binary reader, writer, and error type. Text and
vocabulary loaders can be added here once their data contracts are defined.

## Public API

The root package exports the encoding facade, vocabulary and special-token
types, diagnostic decoder, model-input contracts, and the `read_bin` I/O types.
Vocabularies are still supplied programmatically; tokenizer-file loaders and
compatibility constructors are not implemented yet.
