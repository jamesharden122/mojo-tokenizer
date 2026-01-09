# mojo-tokenizer

High-performance BPE tokenizer for Mojo — 144M tok/s decoding, 3.1x faster than tiktoken.

## Performance

| Implementation | Decoding | Encoding | vs tiktoken |
|----------------|----------|----------|-------------|
| **mojo-tokenizer** | **144 M/s** | **8.0 M/s** | **3.1x faster** |
| rs-bpe (Rust) | 121 M/s | 10.0 M/s* | 2.6x faster |
| tiktoken (Rust) | 47 M/s | 5.1 M/s | baseline |

*rs-bpe raw BPE only; mojo-tokenizer includes full pretokenization pipeline.

Benchmarked on Apple Silicon (M3 Ultra), sherlock.txt (607KB, 143K tokens), 20 iterations.

**[Full benchmarks and methodology →](https://atsentia.com/blog/fastest-ai-token-decoder-mojo-apple-silicon)**

## Quick Start

```mojo
from mojo_tokenizer import Tokenizer

# Load OpenAI's o200k_base vocabulary (GPT-4o, gpt-oss)
var tokenizer = Tokenizer.from_tiktoken("o200k_base")

# Encode text to tokens
var tokens = tokenizer.encode("Hello, world!")
print(tokens)  # [13225, 11, 2375, 0]

# Decode tokens back to text (144M tok/s)
var text = tokenizer.decode(tokens)
print(text)  # "Hello, world!"
```

## Installation

```bash
git clone https://github.com/atsentia/mojo-tokenizer.git
cd mojo-tokenizer
mojo run bench_comprehensive.mojo  # Verify performance
```

## Supported Formats

| Format | Status | Models |
|--------|--------|--------|
| **o200k_base** | ✓ Verified | gpt-5.2, gpt-oss-120B/20B, GPT-4o |
| cl100k_base | ✓ Verified | GPT-4, ChatGPT |
| HuggingFace BPE | Experimental | Qwen, Llama, Mistral |

## How It Works

**FlatTokenStorage** — Decoding at 144M tok/s via contiguous byte array:
```
// Traditional: 100K pointer dereferences
// FlatTokenStorage: Series of memcpy() calls
memcpy(dest, flat_data + offsets[token_id], lengths[token_id])
```

**O(n) Backtracking BPE** — Single-pass encoding with precomputed merge tables (ported from rs-bpe).

**PairCache1000** — O(1) array lookup for common token pairs (+21% encoding speedup).

## License

Apache 2.0

## Links

- [Blog post with full benchmarks](https://atsentia.com/blog/fastest-ai-token-decoder-mojo-apple-silicon)
- [atsentia/mojo-contrib](https://github.com/atsentia/mojo-contrib) — Enterprise Mojo libraries
- [tiktoken](https://github.com/openai/tiktoken) — OpenAI's tokenizer (reference)
- [rs-bpe](https://github.com/rust-ml/rust-gems/tree/main/crates/bpe-openai) — Rust BPE (reference)
