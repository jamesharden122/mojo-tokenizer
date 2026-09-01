# mojo-tokenizer

An explicit, pure-Mojo BPE tokenizer core with allocation-backed tensor staging
for ONNX integration. Vocabularies and merge rules are supplied
programmatically; tokenizer-file loaders and final ONNX `TensorProto`
serialization are not implemented yet.

## Architecture

```text
text
  -> special-token and byte-boundary scanning
  -> BPE encoder
  -> token IDs and optional UTF-8 byte spans
  -> fixed-shape ONNX tensor staging
```

- `core` owns vocabulary storage, special tokens, BPE encoding, diagnostic
  decoding, and private performance structures.
- `preprocessing` contains the internal byte-boundary helpers used by the BPE
  encoder; it is not a separate public pipeline.
- `onnx_utils` copies fixed-shape scalar spans into contiguous allocations for
  batched or single-vector ONNX inputs. Protobuf serialization remains planned.
- `training` constructs deterministic byte-level BPE vocabularies from
  caller-provided strings.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for module boundaries.

## Usage

```mojo
from mojo_tokenizer import BpeTokenizer, SpecialTokenSet, Vocabulary

var vocabulary = Vocabulary()
vocabulary.add_token("a", 0)
vocabulary.add_token("b", 1)
vocabulary.add_token("ab", 2)
vocabulary.add_merge("ab", 0)

var special_tokens = SpecialTokenSet()
special_tokens.add("<eos>", 10)

var tokenizer = BpeTokenizer(vocabulary, special_tokens)
var token_ids = tokenizer.encode("ab<eos>")
var token_spans = tokenizer.encode_with_spans("ab<eos>")
```

### ONNX tensor staging

`OnnxTens[SourceRows, BatchSize, NumCols, dtype]` wraps a caller-owned scalar
span without copying it. The source allocation must remain alive until the
write completes.

- `write_tensor()` copies complete `[BatchSize, NumCols]` views into a new
  contiguous `[SourceRows, NumCols]` allocation. Batches write disjoint row
  ranges in parallel.
- `write_vector()` is the single-request path. It requires `SourceRows == 1`
  and `BatchSize == 1`, uses a `[1, NumCols]` row view, and returns a contiguous
  vector allocation.
- The returned allocation belongs to the caller and must be released with
  `dealloc(output^)`.

Batch mode is selected by the default constructor argument. Pass `False` for
row mode:

```mojo
from mojo_tokenizer import OnnxTens
from std.memory.alloc import dealloc

var batched = OnnxTens[4, 2, 8, DType.int64](values.unsafe_span())
var batch_output = batched.write_tensor()

var vector = OnnxTens[1, 1, 8, DType.int64](values.unsafe_span(), False)
var vector_output = vector.write_vector()

dealloc(batch_output^)
dealloc(vector_output^)
```

See `examples/query_tokenize_pipeline_to_onnx.mojo` for an executable example
that checks both paths.

### Planned ONNX artifacts

The long-term ONNX interface separates two independently usable
`onnx.ModelProto` messages:

```text
raw UTF-8 bytes -> tokenizer ModelProto -> token tensors
token tensors   -> embedding ModelProto -> embeddings
```

The tokenizer model will remain a standalone artifact. An optional project
bundle will contain both the tokenizer and embedding models, together with
explicit bindings from tokenizer outputs (for example, `token_ids` and
`attention_mask`) to embedding-model inputs. The bundle is a project-specific
protobuf container rather than an ONNX `ModelProto`; consumers can extract and
run either contained ONNX model with an ONNX runtime.

The intended `onnx_utils` boundaries are:

```text
src/onnx_utils/
  tokenizer_model.mojo   standalone tokenizer ModelProto construction
  model_bundle.mojo      two-ModelProto bundle serialization
  tensor_binding.mojo    tokenizer-output to model-input mappings
  validation.mojo        type, shape, name, and vocabulary checks
  protobuf/              generic protobuf wire-format support
```

The current `OnnxTens` API stages typed input values only. It does not yet
encode `TensorProto.raw_data`, construct a reusable tokenizer `ModelProto`, or
write an `.onnx` file.

Declared byte-range patterns can emit an existing canonical vocabulary ID
before ordinary BPE runs:

```mojo
from mojo_tokenizer import BpeTokenizer, BytePatternSet
from std.collections.interval import Interval

vocabulary.add_token("<digit>", 3)
var digit = List[Interval[Int]]()
digit.append(Interval(48, 58))
var patterns = BytePatternSet()
patterns.add(digit, 3)

var ranged_tokenizer = BpeTokenizer(vocabulary, special_tokens, patterns)
var ranged_spans = ranged_tokenizer.encode_with_spans("7")
```

Spans are half-open UTF-8 byte offsets. Pattern IDs are intentionally
non-reversible: decoding the example ID produces `<digit>`, not the source
digit.

## Commands

```bash
pixi run build
pixi run test
pixi run example
pixi run benchmark-trie
```

Apache 2.0
