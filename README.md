# mojo-tokenizer

Pure Mojo tokenizer for LLM inference — fast, dependency-free, compatible with tiktoken and HuggingFace formats.

## Why mojo-tokenizer?

MAX Engine currently uses Python-wrapped HuggingFace tokenizers. This library provides a pure Mojo alternative:

- **No Python overhead** — Direct compilation to native code
- **Single binary deployment** — No interpreter or external dependencies
- **Format compatibility** — Load tiktoken (OpenAI) and HuggingFace vocabularies
- **Production ready** — Special token handling, batch processing, chat templates
- **High performance** — 3M+ tokens/sec with optimized caching (94%+ hit rate)

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

# Check cache performance
print("Cache hit rate:", tokenizer.cache_hit_rate())

# Batch processing (benefits from cache warming)
var texts = List[String]()
texts.append("First sentence")
texts.append("Second sentence")
var batch_tokens = tokenizer.encode_batch(texts)
```

## Chat Templates

Format conversations for different LLM models:

```mojo
from mojo_tokenizer.chat import ChatMessage, llama3_template, apply_chat_template

var messages = List[ChatMessage]()
messages.append(ChatMessage.system("You are a helpful assistant."))
messages.append(ChatMessage.user("Hello!"))
messages.append(ChatMessage.assistant("Hi! How can I help you today?"))
messages.append(ChatMessage.user("What's the weather?"))

# Apply Llama 3 format
var formatted = apply_chat_template(messages, llama3_template())
print(formatted)
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
#
# You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>
# ...
```

Supported formats: ChatML, Llama 2, Llama 3, Mistral, Alpaca, Vicuna, Phi-3, Gemma, Zephyr

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
| Tiktoken | ✓ v0.1 | GPT-3.5, GPT-4, GPT-4o |
| HuggingFace JSON | ✓ v0.2 | Llama, Mistral, most HF models |
| SentencePiece | Planned | T5, mT5, multilingual models |

### Pipeline Stages (HuggingFace Compatible)

```mojo
from mojo_tokenizer.pipeline import (
    NormalizerSequence,
    WhitespacePreTokenizer,
    ByteLevelPreTokenizer,
)

# Build a normalizer pipeline
var normalizer = NormalizerSequence()
normalizer.add_lowercase()
normalizer.add_strip()
normalizer.add_whitespace(collapse=True)

var text = normalizer.normalize("  HELLO   WORLD  ")
# Result: "hello world"

# Pre-tokenize for BPE
var pretok = ByteLevelPreTokenizer(add_prefix_space=True)
var tokens = pretok.pre_tokenize("Hello world")
# Result: ["Hello", "Ġworld"]
```

### Caching for Performance

```mojo
# Cache is enabled by default (10k entries)
var tokenizer = BPETokenizer.from_tiktoken("vocab.tiktoken")

# Encode some text (populates cache)
var tokens1 = tokenizer.encode("The quick brown fox")
var tokens2 = tokenizer.encode("The quick brown dog")  # "The", "quick", "brown" cached!

# Check cache statistics
var stats = tokenizer.cache_stats()  # (hits, misses, size)
print("Hit rate:", tokenizer.cache_hit_rate())  # ~80% for natural language

# Cache management
tokenizer.clear_cache()
tokenizer.set_cache_enabled(False)  # Disable for benchmarking
```

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
| `cache_hit_rate()` | Get cache hit rate (0.0-1.0) |
| `cache_stats()` | Get (hits, misses, size) |
| `clear_cache()` | Clear the token cache |
| `set_cache_enabled(bool)` | Enable/disable caching |

### Chat Templates

| Template | Function | Models |
|----------|----------|--------|
| ChatML | `chatml_template()` | GPT-4, Claude, Qwen |
| Llama 2 | `llama2_template()` | Llama 2 Chat |
| Llama 3 | `llama3_template()` | Llama 3 Instruct |
| Mistral | `mistral_template()` | Mistral/Mixtral |
| Alpaca | `alpaca_template()` | Stanford Alpaca |
| Vicuna | `vicuna_template()` | Vicuna |
| Phi-3 | `phi3_template()` | Microsoft Phi-3 |
| Gemma | `gemma_template()` | Google Gemma |
| Zephyr | `zephyr_template()` | HuggingFace Zephyr |

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

## Performance

Benchmarked on M3 Ultra (607KB Sherlock Holmes text):

| Metric | Achieved | Notes |
|--------|----------|-------|
| Cold cache | 6.5M tok/s | First encode of large file |
| Warm cache | 6.0M tok/s | Repeated encoding |
| 5-run avg | 6.2M tok/s | Sustained throughput |
| Cache hit rate | 92%+ | Natural language text |
| Memory (vocab) | <10MB | Loaded vocabulary |
| Startup time | ~100ms | Cold start |

**Exceeds tiktoken (5.2M tok/s)** — Pure Mojo implementation outperforms Rust!

### Optimization History

- **v0.3.1**: Bulk cache eviction (256x speedup for large files)
- **v0.4.0 Phase 1**: Zero-allocation core (2.2x additional speedup)

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

- **v0.1** ✓: BPE core, tiktoken loading, special tokens
- **v0.2** ✓: HuggingFace JSON loading, chat templates, caching, pipeline stages
- **v0.3.1** ✓: LRU cache optimization (256x speedup for large files)
- **v0.4.0** ✓ (current): Zero-allocation core (6.2M tok/s, exceeds tiktoken!)
- **v0.5**: Byte trie for direct lookup, SIMD acceleration (targeting 8M+ tok/s)
- **v1.0**: SentencePiece support, training from corpus, streaming

## License

Apache 2.0 with LLVM Exceptions

## Related

- [mojo-contrib](https://github.com/atsentia/mojo-contrib) — Enterprise Mojo libraries
- [MAX Engine](https://www.modular.com/max) — Modular's inference engine
- [tiktoken](https://github.com/openai/tiktoken) — OpenAI's tokenizer (Rust/Python)
- [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers) — Fast tokenizers (Rust)
