# mojo-tokenizer

Pure Mojo tokenizer for LLM inference — fast, dependency-free, compatible with tiktoken and HuggingFace formats.

## Why mojo-tokenizer?

MAX Engine currently uses Python-wrapped HuggingFace tokenizers. This library provides a pure Mojo alternative:

- **No Python overhead** — Direct compilation to native code
- **Single binary deployment** — No interpreter or external dependencies
- **Format compatibility** — Load tiktoken (OpenAI) and HuggingFace vocabularies
- **Production ready** — Special token handling, batch processing

## Installation

Add to your `pixi.toml`:

```toml
[dependencies]
mojo-tokenizer = { git = "https://github.com/atsentia/mojo-tokenizer" }
```

Or clone directly:

```bash
git clone https://github.com/atsentia/mojo-tokenizer.git
```

## Quick Start

```mojo
from mojo_tokenizer import BPETokenizer

# Load from tiktoken format (OpenAI models)
var tokenizer = BPETokenizer.from_tiktoken("cl100k_base.tiktoken")

# Or load from HuggingFace format
var tokenizer = BPETokenizer.from_huggingface("tokenizer.json")

# Encode text to tokens
var tokens = tokenizer.encode("Hello, world!")
print(tokens)  # [9906, 11, 1917, 0]

# Decode tokens back to text
var text = tokenizer.decode(tokens)
print(text)  # "Hello, world!"

# Batch processing
var texts = List[String]()
texts.append("First sentence")
texts.append("Second sentence")
var batch_tokens = tokenizer.encode_batch(texts)
```

## Features

### BPE Tokenization

Byte Pair Encoding as used by GPT-2, GPT-3, GPT-4, and most modern LLMs:

```mojo
var tokenizer = BPETokenizer()

# The algorithm:
# 1. Convert text to UTF-8 bytes
# 2. Initialize tokens as individual bytes (ids 0-255)
# 3. Iteratively merge highest-priority adjacent pairs
# 4. Return final token IDs
```

### Special Tokens

Handle special tokens that should never be split:

```mojo
# Add special tokens
tokenizer.add_special_token("<|endoftext|>", 50256)
tokenizer.add_special_token("<|im_start|>", 100264)
tokenizer.add_special_token("<|im_end|>", 100265)

# Special tokens are preserved during encoding
var tokens = tokenizer.encode("Hello<|endoftext|>")
# The <|endoftext|> is encoded as a single token, not split
```

### Format Support

| Format | Status | Models |
|--------|--------|--------|
| Tiktoken | v0.1 | GPT-3.5, GPT-4, GPT-4o |
| HuggingFace JSON | v0.2 | Llama, Mistral, most HF models |
| SentencePiece | Planned | T5, mT5, multilingual models |

## API Reference

### BPETokenizer

| Method | Description |
|--------|-------------|
| `from_tiktoken(path)` | Load from tiktoken format |
| `from_huggingface(path)` | Load from HuggingFace tokenizer.json |
| `encode(text)` | Encode text to token IDs |
| `decode(tokens)` | Decode token IDs to text |
| `encode_batch(texts)` | Encode multiple texts |
| `decode_batch(token_lists)` | Decode multiple token lists |
| `vocab_size()` | Get total vocabulary size |
| `add_special_token(text, id)` | Add a special token |

### Token

```mojo
struct Token:
    var id: Int          # Token ID
    var text: String     # Text representation
    var is_special: Bool # Whether this is a special token
```

### Vocabulary

```mojo
struct Vocabulary:
    fn add_token(token: String, id: Int)
    fn get_id(token: String) -> Int
    fn get_text(id: Int) -> String
    fn size() -> Int
```

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Encode throughput | >100k tokens/sec | Single-threaded |
| Memory (vocab) | <10MB | Loaded vocabulary |
| Startup time | <100ms | Cold start |

## Development

### Running Tests

```bash
mojo run tests/test_tokenizer.mojo
```

### Running Examples

```bash
mojo run examples/basic_usage.mojo
```

### Building

```bash
pixi run build
```

## Roadmap

- **v0.1** (current): BPE core, tiktoken loading, special tokens
- **v0.2**: HuggingFace JSON loading, chat templates
- **v0.3**: SentencePiece support, batch optimization
- **v1.0**: Training from corpus, full feature parity

## License

Apache 2.0 with LLVM Exceptions

## Related

- [mojo-contrib](https://github.com/atsentia/mojo-contrib) — Enterprise Mojo libraries
- [MAX Engine](https://www.modular.com/max) — Modular's inference engine
- [tiktoken](https://github.com/openai/tiktoken) — OpenAI's tokenizer (Rust/Python)
- [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers) — Fast tokenizers (Rust)
