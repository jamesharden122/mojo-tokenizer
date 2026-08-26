# mojo-tokenizer

An explicit, pure-Mojo BPE tokenizer core for constructing model-input
pipelines. Vocabularies and merge rules are supplied programmatically. Binary
I/O types are provided by the sibling `read_bin` package; tokenizer-file and
text loaders are not implemented yet.

## Architecture

```text
text
  -> preprocessing
  -> BPE encoder
  -> token IDs
  -> future model-input builder
  -> named Int64 tensors
```

- `core` owns vocabulary storage, special tokens, BPE encoding, diagnostic
  decoding, and private performance structures.
- `preprocessing` owns normalization, pre-tokenization, token-sequence
  post-processing, and scanning primitives.
- `model_input` defines the future typed-buffer contract. Buffer construction,
  padding, truncation, ONNX execution, and SurrealML integration are not yet
  implemented.
- Persistent model-input values use allocator-backed `Allocation[Int64]`
  buffers; small shape metadata and temporary collections continue to use
  `List`.
- `io` exposes `read_bin` binary readers, writers, and structured I/O errors;
  future text and vocabulary readers belong behind this boundary.

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
```

## Commands

```bash
pixi run build
pixi run test
pixi run example
```

Apache 2.0
