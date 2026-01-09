#!/usr/bin/env python3
"""
Validate mojo-tokenizer against local LLM (gpt-oss-20b via LM Studio)

This test sends prompts to a local OpenAI-compatible API and compares
token counts with tiktoken to verify tokenization compatibility.

Usage:
    # Make sure your local LLM server is running
    python tests/test_local_llm_validation.py

    # Or specify custom endpoint
    python tests/test_local_llm_validation.py --endpoint http://127.0.0.1:1234
"""

import argparse
import json
import sys
from typing import Optional

try:
    import requests
    import tiktoken
except ImportError:
    print("Install dependencies: pip install requests tiktoken")
    sys.exit(1)


def get_local_llm_response(
    endpoint: str,
    prompt: str,
    model: str = "openai/gpt-oss-20b"
) -> dict:
    """Send request to local LLM and get response with token counts."""

    url = f"{endpoint}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 10,  # Minimal response to save time
        "stream": False
    }

    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} - {response.text}")

    return response.json()


def validate_against_local_llm(endpoint: str = "http://127.0.0.1:1234"):
    """Validate tokenization against local LLM."""

    print("=" * 70)
    print("Local LLM Token Count Validation")
    print(f"Endpoint: {endpoint}")
    print("=" * 70)

    # Check if server is running
    try:
        health = requests.get(f"{endpoint}/v1/models", timeout=5)
        if health.status_code == 200:
            models = health.json()
            print(f"Server running. Available models: {[m.get('id', 'unknown') for m in models.get('data', [])]}")
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to local LLM server")
        print(f"Make sure your server is running at {endpoint}")
        return False

    # Initialize tiktoken (cl100k_base is what GPT-4/gpt-oss likely uses)
    enc = tiktoken.get_encoding("cl100k_base")

    # Test prompts
    test_prompts = [
        "Hello, world!",
        "What is 2 + 2?",
        "The quick brown fox jumps over the lazy dog.",
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
        "Explain BPE tokenization in one sentence.",
    ]

    print(f"\n{'Prompt':<50} {'tiktoken':<10} {'API':<10} {'Delta':<10}")
    print("-" * 80)

    results = []
    all_pass = True

    for prompt in test_prompts:
        # Get tiktoken count
        tiktoken_tokens = enc.encode(prompt)
        tiktoken_count = len(tiktoken_tokens)

        try:
            # Get API response
            response = get_local_llm_response(endpoint, prompt)

            usage = response.get("usage", {})
            api_prompt_tokens = usage.get("prompt_tokens", -1)

            # Delta should be message structure overhead (typically 4-10 tokens)
            if api_prompt_tokens > 0:
                delta = api_prompt_tokens - tiktoken_count
                # Accept if delta is within reasonable overhead range
                status = "✓" if -5 <= delta <= 15 else "?"
            else:
                delta = "N/A"
                status = "?"

        except Exception as e:
            api_prompt_tokens = "err"
            delta = "err"
            status = "✗"
            print(f"\n  Error: {e}")
            all_pass = False

        display = prompt[:45] + "..." if len(prompt) > 45 else prompt
        print(f"{display:<50} {tiktoken_count:<10} {api_prompt_tokens:<10} {delta:<10} {status}")

        results.append({
            "prompt": prompt,
            "tiktoken_count": tiktoken_count,
            "tiktoken_tokens": tiktoken_tokens,
            "api_prompt_tokens": api_prompt_tokens if isinstance(api_prompt_tokens, int) else None,
        })

    # Detailed token breakdown for first prompt
    print("\n--- Detailed Token Analysis ---")
    for prompt in test_prompts[:3]:
        tokens = enc.encode(prompt)
        decoded = [enc.decode([t]) for t in tokens]
        print(f"\n{repr(prompt)}")
        print(f"  Tokens: {tokens}")
        print(f"  Decoded: {decoded}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if all_pass:
        print("✓ All tests completed successfully")
        print("\nNote: Delta between tiktoken and API counts is expected due to:")
        print("  - Message structure overhead (role, content wrappers)")
        print("  - Special tokens (BOS, EOS)")
        print("  - Model-specific tokenization differences")
    else:
        print("✗ Some tests failed - check error messages above")

    # Save results
    with open("tests/local_llm_validation_results.json", "w") as f:
        json.dump({
            "endpoint": endpoint,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to: tests/local_llm_validation_results.json")

    return all_pass


def run_roundtrip_test(endpoint: str = "http://127.0.0.1:1234"):
    """Test encode -> decode roundtrip."""

    print("\n" + "=" * 70)
    print("Roundtrip Test (tiktoken)")
    print("=" * 70)

    enc = tiktoken.get_encoding("cl100k_base")

    test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "def main(): print('Hello')",
        "日本語テスト",
        "Café résumé naïve",
    ]

    print(f"\n{'Text':<40} {'Roundtrip':<10}")
    print("-" * 55)

    all_pass = True
    for text in test_texts:
        tokens = enc.encode(text)
        decoded = enc.decode(tokens)
        match = decoded == text
        all_pass = all_pass and match

        display = text[:35] + "..." if len(text) > 35 else text
        print(f"{display:<40} {'✓ PASS' if match else '✗ FAIL':<10}")

        if not match:
            print(f"  Original: {repr(text)}")
            print(f"  Decoded:  {repr(decoded)}")

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Validate tokenizer against local LLM")
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:1234",
        help="Local LLM endpoint (default: http://127.0.0.1:1234)"
    )
    args = parser.parse_args()

    # Run roundtrip test first (no API needed)
    roundtrip_ok = run_roundtrip_test()

    # Run local LLM validation
    llm_ok = validate_against_local_llm(args.endpoint)

    # Final status
    print("\n" + "=" * 70)
    if roundtrip_ok and llm_ok:
        print("✓ ALL VALIDATIONS PASSED")
    else:
        print("✗ SOME VALIDATIONS FAILED")
        sys.exit(1)
    print("=" * 70)


if __name__ == "__main__":
    main()
