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

Features:
    - Pure Mojo implementation (no Python dependencies)
    - BPE (Byte Pair Encoding) algorithm
    - Tiktoken format support (OpenAI compatible)
    - HuggingFace tokenizer.json support
    - Special token handling
    - Batch encoding/decoding

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
