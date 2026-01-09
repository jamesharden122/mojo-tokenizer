#!/usr/bin/env python3
"""
Comprehensive Tokenizer Validation Suite

Validates mojo-tokenizer against:
1. OpenAI API (tiktoken / cl100k_base)
2. Open-source models (Llama, Mistral, Qwen via HuggingFace)

Usage:
    # Offline validation (no API keys needed)
    python tests/test_tokenizer_validation.py

    # With OpenAI API validation
    export OPENAI_API_KEY="your-key"
    python tests/test_tokenizer_validation.py

    # With HuggingFace token (for gated models like Llama)
    export HF_TOKEN="your-token"
    python tests/test_tokenizer_validation.py
"""

import os
import sys
import json
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

# Test corpus - diverse text samples
TEST_CORPUS = [
    # Simple text
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",

    # Numbers and punctuation
    "Price: $19.99 (20% off!) - Limited time offer.",
    "Version 2.0.1-beta.3 released on 2024-01-15",

    # Code
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "SELECT u.name, COUNT(*) FROM users u GROUP BY u.id HAVING COUNT(*) > 5;",
    "const arr = [1, 2, 3].map(x => x * 2).filter(x => x > 2);",

    # Unicode
    "Café résumé naïve",
    "日本語テスト",
    "Привет мир",
    "🚀 Launch successful! 🎉",

    # Longer text (typical LLM prompt)
    """You are a helpful AI assistant. Please help me understand how tokenization works.

    Tokenization is the process of breaking text into smaller units called tokens.
    Different models use different tokenization strategies:
    - BPE (Byte Pair Encoding): Used by GPT, Llama, Mistral
    - WordPiece: Used by BERT
    - SentencePiece: Used by T5, mT5

    Can you explain the BPE algorithm?""",

    # JSON
    '{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}',

    # Markdown
    """# BPE Tokenizer

## Overview

**Byte Pair Encoding** is an algorithm that:

1. Starts with individual bytes
2. Iteratively merges the most frequent pairs
3. Builds a vocabulary of subword units

```python
def encode(text):
    tokens = list(text.encode('utf-8'))
    while can_merge(tokens):
        tokens = merge_best_pair(tokens)
    return tokens
```
""",
]


@dataclass
class ValidationResult:
    model_name: str
    text: str
    expected_count: int
    expected_tokens: Optional[List[int]]
    notes: str = ""


def validate_tiktoken():
    """Validate against tiktoken (OpenAI's tokenizer)."""
    try:
        import tiktoken
    except ImportError:
        print("  Skipping tiktoken validation (pip install tiktoken)")
        return []

    print("\n" + "=" * 70)
    print("OpenAI / tiktoken Validation (cl100k_base)")
    print("=" * 70)

    enc = tiktoken.get_encoding("cl100k_base")
    results = []

    print(f"\n{'Text':<50} {'Tokens':<10}")
    print("-" * 70)

    for text in TEST_CORPUS:
        tokens = enc.encode(text)
        count = len(tokens)

        display = text[:45].replace("\n", "\\n") + ("..." if len(text) > 45 else "")
        print(f"{display:<50} {count:<10}")

        results.append(ValidationResult(
            model_name="tiktoken/cl100k_base",
            text=text,
            expected_count=count,
            expected_tokens=tokens,
        ))

    # Show some token examples
    print("\n--- Token Examples ---")
    for text in ["Hello, world!", "def main():", "🚀"]:
        tokens = enc.encode(text)
        decoded = [enc.decode([t]) for t in tokens]
        print(f"{repr(text):20} -> {tokens} -> {decoded}")

    return results


def validate_huggingface_models():
    """Validate against HuggingFace open-source models."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("  Skipping HuggingFace validation (pip install transformers)")
        return []

    # Models to test (BPE-based, tiktoken-compatible architecture)
    models = [
        # ("meta-llama/Llama-2-7b-hf", "Llama 2"),  # Requires HF_TOKEN
        # ("meta-llama/Meta-Llama-3-8B", "Llama 3"),  # Requires HF_TOKEN
        ("mistralai/Mistral-7B-v0.1", "Mistral 7B"),
        ("Qwen/Qwen2-1.5B", "Qwen 2"),
        ("microsoft/phi-2", "Phi-2"),
    ]

    hf_token = os.environ.get("HF_TOKEN")
    all_results = []

    for model_id, model_name in models:
        print("\n" + "=" * 70)
        print(f"{model_name} Validation ({model_id})")
        print("=" * 70)

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=hf_token,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"  Skipping {model_name}: {e}")
            continue

        print(f"  Vocab size: {tokenizer.vocab_size}")
        print(f"  Tokenizer type: {type(tokenizer).__name__}")

        print(f"\n{'Text':<50} {'Tokens':<10}")
        print("-" * 70)

        results = []
        for text in TEST_CORPUS[:5]:  # Fewer tests for HF models (slower)
            tokens = tokenizer.encode(text, add_special_tokens=False)
            count = len(tokens)

            display = text[:45].replace("\n", "\\n") + ("..." if len(text) > 45 else "")
            print(f"{display:<50} {count:<10}")

            results.append(ValidationResult(
                model_name=model_id,
                text=text,
                expected_count=count,
                expected_tokens=tokens,
            ))

        all_results.extend(results)

        # Show token examples
        print("\n--- Token Examples ---")
        text = "Hello, world!"
        tokens = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.convert_ids_to_tokens(tokens)
        print(f"{repr(text):20} -> {tokens[:10]} -> {decoded[:10]}")

    return all_results


def validate_openai_api():
    """Validate against actual OpenAI API responses."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n  Skipping OpenAI API validation (set OPENAI_API_KEY)")
        return []

    try:
        from openai import OpenAI
        import tiktoken
    except ImportError:
        print("  Skipping OpenAI API validation (pip install openai tiktoken)")
        return []

    print("\n" + "=" * 70)
    print("OpenAI API Live Validation")
    print("=" * 70)

    client = OpenAI(api_key=api_key)
    enc = tiktoken.encoding_for_model("gpt-4o")

    # Test a few prompts against the live API
    test_texts = [
        "Hello, how are you?",
        "Explain quantum computing in simple terms.",
        "def quicksort(arr): return arr if len(arr) <= 1 else quicksort([x for x in arr[1:] if x < arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x >= arr[0]])",
    ]

    print(f"\n{'Text':<40} {'tiktoken':<10} {'API':<10} {'Delta'}")
    print("-" * 70)

    results = []
    for text in test_texts:
        tiktoken_count = len(enc.encode(text))

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": text}],
                max_tokens=1,
            )
            api_count = response.usage.prompt_tokens

            # API count includes message structure overhead (~4-7 tokens)
            # So we check if delta is within expected overhead range
            delta = api_count - tiktoken_count
            status = "✓" if 0 <= delta <= 10 else "?"

        except Exception as e:
            api_count = -1
            delta = "err"
            status = "✗"
            print(f"  Error: {e}")

        display = text[:35] + "..." if len(text) > 35 else text
        print(f"{display:<40} {tiktoken_count:<10} {api_count:<10} {delta}")

        results.append(ValidationResult(
            model_name="openai-api",
            text=text,
            expected_count=tiktoken_count,
            expected_tokens=None,
            notes=f"API returned {api_count} (delta={delta})"
        ))

    return results


def save_reference_tokens(results: List[ValidationResult], output_path: str):
    """Save reference tokens for Mojo validation."""
    data = {
        "description": "Reference token counts for mojo-tokenizer validation",
        "models": {},
    }

    for r in results:
        if r.model_name not in data["models"]:
            data["models"][r.model_name] = []

        data["models"][r.model_name].append({
            "text": r.text,
            "token_count": r.expected_count,
            "tokens": r.expected_tokens,
        })

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved reference tokens to: {output_path}")


def main():
    print("=" * 70)
    print("Tokenizer Validation Suite")
    print("=" * 70)
    print("\nValidating against multiple tokenizer implementations...")

    all_results = []

    # 1. tiktoken (OpenAI's tokenizer)
    all_results.extend(validate_tiktoken())

    # 2. HuggingFace open-source models
    all_results.extend(validate_huggingface_models())

    # 3. Live OpenAI API
    all_results.extend(validate_openai_api())

    # Save reference data for Mojo tests
    if all_results:
        save_reference_tokens(
            all_results,
            "tests/reference_tokens.json"
        )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    models_tested = set(r.model_name for r in all_results)
    print(f"Models validated: {len(models_tested)}")
    for model in sorted(models_tested):
        count = sum(1 for r in all_results if r.model_name == model)
        print(f"  - {model}: {count} test cases")

    print(f"\nTotal test cases: {len(all_results)}")
    print("\nTo use these reference tokens in Mojo tests:")
    print("  1. Load tests/reference_tokens.json")
    print("  2. Compare mojo-tokenizer output with expected_tokens")
    print("  3. Verify exact match for tiktoken/cl100k_base")


if __name__ == "__main__":
    main()
