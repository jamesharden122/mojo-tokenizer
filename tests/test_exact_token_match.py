#!/usr/bin/env python3
"""
Exact Token Match Test

The definitive validation: mojo-tokenizer MUST produce identical token
sequences to tiktoken. If tokens match exactly, the model behavior will
be identical regardless of which tokenizer is used.

This generates reference data that can be used to validate the Mojo implementation.

Usage:
    python tests/test_exact_token_match.py
"""

import json
import sys

try:
    import tiktoken
except ImportError:
    print("Install tiktoken: pip install tiktoken")
    sys.exit(1)


# Comprehensive test corpus covering edge cases
TEST_CORPUS = {
    "simple": [
        "Hello",
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "   spaces   ",
        "",
        " ",
        "\n",
        "\t",
    ],
    "numbers": [
        "123",
        "3.14159",
        "$19.99",
        "1,000,000",
        "-42",
        "0x1F600",
        "1e-10",
    ],
    "punctuation": [
        "Hello, world!",
        "What's up?",
        "He said: \"Hello\"",
        "It's a test...",
        "foo@bar.com",
        "https://example.com/path?q=1&x=2",
        "(a + b) * c / d - e",
    ],
    "code": [
        "def hello():",
        "def hello(): pass",
        "print('Hello, World!')",
        "for i in range(10):",
        "if x > 0 and y < 10:",
        "lambda x: x * 2",
        "class Foo(Bar):",
        "# comment",
        "/* multi-line */",
        "x = [1, 2, 3]",
        "d = {'a': 1, 'b': 2}",
        "f\"Hello {name}\"",
        "SELECT * FROM users WHERE id = 1;",
        "const x = () => {};",
    ],
    "unicode": [
        "café",
        "naïve",
        "résumé",
        "日本語",
        "中文测试",
        "Привет мир",
        "مرحبا",
        "שלום",
        "🚀",
        "👍🏽",
        "🇺🇸",
        "Hello 🌍 World",
        "Math: α + β = γ",
        "∑∏∫∂",
        "→←↑↓",
    ],
    "whitespace": [
        "a b",
        "a  b",
        "a   b",
        "a\tb",
        "a\nb",
        "a\r\nb",
        "  leading",
        "trailing  ",
        "  both  ",
    ],
    "special_sequences": [
        "<|endoftext|>",  # Should NOT be split (special token)
        "<s>",
        "</s>",
        "[INST]",
        "[/INST]",
        "<<SYS>>",
        "<</SYS>>",
    ],
    "long_text": [
        "The quick brown fox jumps over the lazy dog. " * 10,
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 5,
    ],
    "edge_cases": [
        "\\n",  # Escaped newline (2 chars, not newline)
        "\\t",  # Escaped tab
        "\\\\",  # Double backslash
        "'",
        "\"",
        "`",
        "'''",
        '"""',
        "a" * 100,  # Repeated char
        "ab" * 50,  # Repeated pair
    ],
}


def generate_reference_tokens():
    """Generate reference token data for all test cases."""

    enc = tiktoken.get_encoding("cl100k_base")

    print("=" * 70)
    print("Generating Reference Tokens (cl100k_base)")
    print("=" * 70)

    reference_data = {
        "encoding": "cl100k_base",
        "vocab_size": enc.n_vocab,
        "categories": {},
    }

    total_tests = 0
    total_tokens = 0

    for category, texts in TEST_CORPUS.items():
        print(f"\n--- {category} ---")
        reference_data["categories"][category] = []

        for text in texts:
            tokens = enc.encode(text, allowed_special="all")
            decoded = enc.decode(tokens)
            roundtrip_ok = decoded == text

            entry = {
                "text": text,
                "tokens": tokens,
                "token_count": len(tokens),
                "roundtrip_ok": roundtrip_ok,
            }
            reference_data["categories"][category].append(entry)

            total_tests += 1
            total_tokens += len(tokens)

            # Display
            display = repr(text)[:40]
            status = "✓" if roundtrip_ok else "✗"
            print(f"  {display:<42} -> {len(tokens):3} tokens {status}")

            if not roundtrip_ok:
                print(f"    WARNING: Roundtrip failed!")
                print(f"    Original: {repr(text)}")
                print(f"    Decoded:  {repr(decoded)}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total test cases: {total_tests}")
    print(f"Total tokens: {total_tokens}")
    print(f"Categories: {len(TEST_CORPUS)}")

    # Save reference data
    output_path = "tests/reference_tokens_cl100k.json"
    with open(output_path, "w") as f:
        json.dump(reference_data, f, indent=2, ensure_ascii=False)

    print(f"\nReference data saved to: {output_path}")
    print("\nThis file contains the EXACT tokens that mojo-tokenizer must produce.")
    print("Any deviation means incorrect tokenization.")

    return reference_data


def show_token_examples():
    """Show detailed token breakdowns for common cases."""

    enc = tiktoken.get_encoding("cl100k_base")

    print("\n" + "=" * 70)
    print("Token Breakdown Examples")
    print("=" * 70)

    examples = [
        "Hello, world!",
        "The quick brown fox",
        "def main(): pass",
        "https://example.com",
        "日本語",
        "🚀",
    ]

    for text in examples:
        tokens = enc.encode(text)
        decoded_parts = [enc.decode([t]) for t in tokens]

        print(f"\nText: {repr(text)}")
        print(f"Tokens: {tokens}")
        print(f"Parts: {decoded_parts}")
        print(f"Bytes: {[t.to_bytes(4, 'big') for t in tokens]}")


def create_mojo_test_file():
    """Generate a Mojo test file that validates against reference tokens."""

    mojo_test = '''"""
Auto-generated test: Validate mojo-tokenizer against reference tokens.

Run: mojo run tests/test_reference_tokens.mojo
"""

from testing import assert_equal
from collections import List
import json


fn main() raises:
    print("Loading reference tokens...")

    # Load reference data
    with open("tests/reference_tokens_cl100k.json", "r") as f:
        var data = json.load(f)

    print("Running exact token match tests...")

    var passed = 0
    var failed = 0

    # TODO: Load mojo-tokenizer and compare
    # var tokenizer = BPETokenizer.from_tiktoken("data/cl100k_base.tiktoken")

    # Example structure:
    # for category in data["categories"]:
    #     for test_case in category:
    #         var text = test_case["text"]
    #         var expected = test_case["tokens"]
    #         var actual = tokenizer.encode(text)
    #         if actual == expected:
    #             passed += 1
    #         else:
    #             failed += 1
    #             print("FAIL:", text)
    #             print("  Expected:", expected)
    #             print("  Actual:", actual)

    print("Passed:", passed)
    print("Failed:", failed)

    if failed > 0:
        raise Error("Token mismatch detected!")
'''

    with open("tests/test_reference_tokens.mojo", "w") as f:
        f.write(mojo_test)

    print("\nGenerated: tests/test_reference_tokens.mojo")


def main():
    # Generate reference tokens
    generate_reference_tokens()

    # Show examples
    show_token_examples()

    # Create Mojo test template
    create_mojo_test_file()

    print("\n" + "=" * 70)
    print("VALIDATION APPROACH")
    print("=" * 70)
    print("""
To validate mojo-tokenizer:

1. Load tests/reference_tokens_cl100k.json
2. For each test case:
   - Encode text with mojo-tokenizer
   - Compare token list with reference
   - MUST be EXACT match (same tokens, same order)

3. If any mismatch:
   - The tokenization is WRONG
   - Model behavior will differ from tiktoken
   - Debug the specific case

This is the definitive test - no approximations, no deltas.
Tokens must match EXACTLY.
""")


if __name__ == "__main__":
    main()
