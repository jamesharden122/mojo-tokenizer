#!/usr/bin/env python3
"""
HuggingFace BPE Validation Test

Validates mojo-tokenizer against HuggingFace transformers tokenizers.
Tests both correctness (exact token match) and performance.

Usage:
    python tests/test_huggingface_validation.py
"""

import json
import os
import sys
import time
from pathlib import Path

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Install transformers: pip install transformers")
    sys.exit(1)


# Test corpus - same as tiktoken validation
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

    # Longer text
    """You are a helpful AI assistant. Please help me understand how tokenization works.
    Tokenization is the process of breaking text into smaller units called tokens.""",

    # JSON
    '{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}',

    # Whitespace edge cases
    "   spaces   ",
    "a\tb\nc",
    "",
    " ",
]

# Models to test (BPE-based, publicly accessible)
MODELS_TO_TEST = [
    ("Qwen/Qwen2-1.5B", "Qwen 2"),
    ("mistralai/Mistral-7B-v0.1", "Mistral 7B"),
    # ("meta-llama/Llama-2-7b-hf", "Llama 2"),  # Requires HF_TOKEN
]


def download_tokenizer(model_id: str, output_dir: Path) -> Path:
    """Download tokenizer.json from HuggingFace."""
    print(f"  Downloading {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Save to local directory
    model_dir = output_dir / model_id.replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(str(model_dir))

    tokenizer_json = model_dir / "tokenizer.json"
    if tokenizer_json.exists():
        print(f"  Saved to: {tokenizer_json}")
        return tokenizer_json
    else:
        print(f"  Warning: No tokenizer.json found (might use tokenizer.model)")
        return None


def generate_reference_tokens(model_id: str, tokenizer) -> dict:
    """Generate reference tokens for all test cases."""
    results = {
        "model_id": model_id,
        "vocab_size": tokenizer.vocab_size,
        "test_cases": []
    }

    for text in TEST_CORPUS:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(tokens)

        results["test_cases"].append({
            "text": text,
            "tokens": tokens,
            "token_count": len(tokens),
            "roundtrip_ok": decoded.strip() == text.strip(),
        })

    return results


def benchmark_tokenizer(tokenizer, text: str, iterations: int = 100) -> dict:
    """Benchmark encoding/decoding performance."""
    # Warmup
    for _ in range(10):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokenizer.decode(tokens)

    # Encoding benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        tokens = tokenizer.encode(text, add_special_tokens=False)
    encode_time = time.perf_counter() - start

    # Decoding benchmark
    tokens = tokenizer.encode(text, add_special_tokens=False)
    start = time.perf_counter()
    for _ in range(iterations):
        tokenizer.decode(tokens)
    decode_time = time.perf_counter() - start

    token_count = len(tokens)

    return {
        "iterations": iterations,
        "token_count": token_count,
        "encode_time_ms": encode_time * 1000,
        "decode_time_ms": decode_time * 1000,
        "encode_tok_per_sec": (token_count * iterations) / encode_time,
        "decode_tok_per_sec": (token_count * iterations) / decode_time,
    }


def run_validation():
    """Run full validation suite."""
    print("=" * 70)
    print("HuggingFace BPE Validation Suite")
    print("=" * 70)

    # Setup directories
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "huggingface"
    data_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for model_id, model_name in MODELS_TO_TEST:
        print(f"\n{'='*70}")
        print(f"Testing: {model_name} ({model_id})")
        print("=" * 70)

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            print(f"  Vocab size: {tokenizer.vocab_size}")
            print(f"  Tokenizer class: {type(tokenizer).__name__}")

            # Download tokenizer.json for mojo-tokenizer
            tokenizer_json = download_tokenizer(model_id, data_dir)

            # Generate reference tokens
            print("\n--- Correctness Test ---")
            ref_tokens = generate_reference_tokens(model_id, tokenizer)

            passed = 0
            failed = 0
            for tc in ref_tokens["test_cases"]:
                status = "✓" if tc["roundtrip_ok"] else "✗"
                if tc["roundtrip_ok"]:
                    passed += 1
                else:
                    failed += 1

                display = repr(tc["text"])[:40]
                print(f"  {display:<42} {tc['token_count']:3} tokens {status}")

            print(f"\n  Passed: {passed}/{passed+failed}")

            # Save reference tokens
            ref_file = data_dir / f"{model_id.replace('/', '_')}_reference.json"
            with open(ref_file, "w") as f:
                json.dump(ref_tokens, f, indent=2, ensure_ascii=False)
            print(f"  Reference saved: {ref_file}")

            # Performance benchmark
            print("\n--- Performance Benchmark ---")

            # Use sherlock.txt if available, otherwise use long test text
            sherlock_path = base_dir / "data" / "sherlock.txt"
            if sherlock_path.exists():
                with open(sherlock_path) as f:
                    bench_text = f.read()
                print(f"  Using: sherlock.txt ({len(bench_text):,} chars)")
            else:
                bench_text = TEST_CORPUS[-4] * 100  # Repeat long text
                print(f"  Using: repeated test text ({len(bench_text):,} chars)")

            perf = benchmark_tokenizer(tokenizer, bench_text, iterations=20)

            print(f"  Encoding: {perf['encode_tok_per_sec']/1e6:.2f} M tok/s")
            print(f"  Decoding: {perf['decode_tok_per_sec']/1e6:.2f} M tok/s")
            print(f"  Token count: {perf['token_count']:,}")

            all_results[model_id] = {
                "correctness": ref_tokens,
                "performance": perf,
                "tokenizer_json": str(tokenizer_json) if tokenizer_json else None,
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[model_id] = {"error": str(e)}

    # Save all results
    results_file = data_dir / "validation_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\nAll results saved to: {results_file}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<30} {'Vocab':<10} {'Encode':<15} {'Decode':<15}")
    print("-" * 70)

    for model_id, result in all_results.items():
        if "error" in result:
            print(f"{model_id:<30} ERROR: {result['error'][:30]}")
        else:
            vocab = result["correctness"]["vocab_size"]
            enc = result["performance"]["encode_tok_per_sec"] / 1e6
            dec = result["performance"]["decode_tok_per_sec"] / 1e6
            print(f"{model_id:<30} {vocab:<10} {enc:.2f} M/s       {dec:.2f} M/s")

    print("\n" + "=" * 70)
    print("Next: Run mojo-tokenizer on the same tokenizer.json files")
    print("and compare tokens + performance")
    print("=" * 70)

    return all_results


def show_token_examples():
    """Show detailed token breakdowns."""
    print("\n" + "=" * 70)
    print("Token Examples")
    print("=" * 70)

    for model_id, model_name in MODELS_TO_TEST[:1]:  # Just first model
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

            print(f"\n{model_name}:")
            for text in ["Hello, world!", "def main():", "🚀"]:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                decoded = [tokenizer.decode([t]) for t in tokens]
                print(f"  {repr(text):20} -> {tokens} -> {decoded}")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    results = run_validation()
    show_token_examples()
