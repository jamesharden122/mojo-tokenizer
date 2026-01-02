"""
Mojo Tokenizer Library

Fast, pure Mojo tokenization for LLM inference pipelines.
Supports BPE encoding with tiktoken and HuggingFace format compatibility.

Basic Usage:
    from mojo_tokenizer import BPETokenizer

    # Load a tiktoken vocabulary
    var tokenizer = BPETokenizer.from_tiktoken("path/to/vocab.tiktoken")

    # Encode text to tokens
    var tokens = tokenizer.encode("Hello, world!")

    # Decode tokens back to text
    var text = tokenizer.decode(tokens)

    # Check cache performance
    print("Cache hit rate:", tokenizer.cache_hit_rate())

Chat Templates:
    from mojo_tokenizer.chat import ChatMessage, llama3_template, apply_chat_template

    var messages = List[ChatMessage]()
    messages.append(ChatMessage.system("You are a helpful assistant."))
    messages.append(ChatMessage.user("Hello!"))

    var formatted = apply_chat_template(messages, llama3_template())

Features:
    - Pure Mojo implementation (no Python dependencies)
    - BPE (Byte Pair Encoding) algorithm
    - Tiktoken format support (OpenAI compatible)
    - HuggingFace tokenizer.json support
    - Special token handling
    - Batch encoding/decoding
    - Word-level LRU caching (80%+ hit rate)
    - SIMD-optimized string operations
    - Chat templates (Llama 2/3, Mistral, ChatML, etc.)
    - Pipeline stages (normalizer, pre-tokenizer, post-processor)

Performance:
    - 100k+ tokens/sec on M3 Ultra
    - <100ms vocabulary loading
    - Move semantics for zero-copy operations

Part of mojo-contrib: https://github.com/atsentia/mojo-contrib
"""

# Core types
from .tokenizer import Tokenizer, Token

# BPE implementation
from .bpe import BPETokenizer

# Vocabulary management
from .vocab import Vocabulary, MergeRule

# Special tokens
from .special_tokens import SpecialTokens, SpecialToken

# Format loaders
from .formats import load_tiktoken, load_huggingface

# Chat templates
from .chat import ChatTemplate, ChatMessage, apply_chat_template
from .chat.formats import (
    chatml_template,
    llama2_template,
    llama3_template,
    mistral_template,
    alpaca_template,
    vicuna_template,
)

# Pipeline stages
from .pipeline import (
    Normalizer,
    NormalizerSequence,
    PreTokenizer,
    WhitespacePreTokenizer,
    ByteLevelPreTokenizer,
    PostProcessor,
    TemplatePostProcessor,
)

# Caching
from .cache import TokenCache, MergeCache

# Benchmarking
from .benchmark import BenchmarkRunner, BenchmarkResult, run_benchmark
