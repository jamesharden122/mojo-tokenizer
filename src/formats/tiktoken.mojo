"""
Tiktoken format loader.

Tiktoken is OpenAI's tokenizer format used by GPT-3.5, GPT-4, and other
OpenAI models. The format stores BPE merge rules as base64-encoded tokens
with their ranks.

File format (each line):
    <base64-encoded-token> <rank>

Example:
    IQ== 0
    Ig== 1
    Iw== 2
"""

from ..vocab import Vocabulary
from ..special_tokens import SpecialTokens


fn load_tiktoken(path: String) raises -> (Vocabulary, SpecialTokens):
    """
    Load a tiktoken vocabulary file.

    Args:
        path: Path to the .tiktoken file.

    Returns:
        Tuple of (Vocabulary, SpecialTokens).

    Raises:
        Error if file cannot be read or parsed.

    File Format:
        Each line contains a base64-encoded token and its rank,
        separated by a space. The rank determines merge priority
        (lower = higher priority).

    Example file contents:
        IQ== 0
        Ig== 1
        Iw== 2
        ...

    Usage:
        var vocab, special = load_tiktoken("cl100k_base.tiktoken")
    """
    var vocab = Vocabulary()
    var special = SpecialTokens()

    # TODO: Implement file reading and base64 decoding
    # The implementation will:
    # 1. Read file line by line
    # 2. Split each line on space
    # 3. Base64 decode the token
    # 4. Add token to vocabulary with rank as ID
    # 5. Build merge rules from consecutive token pairs

    raise Error("tiktoken loading not yet implemented - requires file I/O and base64")


fn load_tiktoken_with_special(
    path: String,
    special_tokens: Dict[String, Int]
) raises -> (Vocabulary, SpecialTokens):
    """
    Load a tiktoken vocabulary with additional special tokens.

    Args:
        path: Path to the .tiktoken file.
        special_tokens: Dict mapping special token text to ID.

    Returns:
        Tuple of (Vocabulary, SpecialTokens).

    Example:
        var special = Dict[String, Int]()
        special["<|endoftext|>"] = 100256
        special["<|im_start|>"] = 100264
        special["<|im_end|>"] = 100265

        var vocab, tokens = load_tiktoken_with_special(
            "cl100k_base.tiktoken",
            special
        )
    """
    var vocab, special = load_tiktoken(path)

    # Add special tokens
    for item in special_tokens.items():
        special.add(item[].key, item[].value)

    return (vocab, special)
