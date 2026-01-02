"""
BPE (Byte Pair Encoding) tokenizer implementation.

This module implements the BPE algorithm used by GPT-2, GPT-3, GPT-4,
and many other large language models. It supports loading vocabularies
from tiktoken and HuggingFace formats.

Algorithm:
    1. Convert text to UTF-8 bytes
    2. Initialize tokens as individual bytes (ids 0-255)
    3. Iteratively merge highest-priority adjacent pairs
    4. Return final token IDs
"""

from .tokenizer import Tokenizer, Token
from .vocab import Vocabulary, MergeRule
from .special_tokens import SpecialTokens


struct BPETokenizer(Tokenizer):
    """
    Byte Pair Encoding tokenizer.

    Supports loading from tiktoken and HuggingFace formats.
    Handles special tokens and provides efficient batch encoding.
    """

    var vocab: Vocabulary
    """The vocabulary mapping tokens to IDs."""

    var special_tokens: SpecialTokens
    """Special tokens configuration."""

    var _byte_encoder: Dict[Int, String]
    """Byte to unicode character mapping."""

    var _byte_decoder: Dict[String, Int]
    """Unicode character to byte mapping."""

    fn __init__(out self):
        """Create an empty BPETokenizer."""
        self.vocab = Vocabulary()
        self.special_tokens = SpecialTokens()
        self._byte_encoder = Dict[Int, String]()
        self._byte_decoder = Dict[String, Int]()
        self._init_byte_mappings()

    fn _init_byte_mappings(mut self):
        """Initialize byte-to-unicode mappings for BPE."""
        # Standard printable ASCII range
        for i in range(ord("!"), ord("~") + 1):
            self._byte_encoder[i] = chr(i)
            self._byte_decoder[chr(i)] = i

        # Extended mappings for non-printable bytes
        var n = 0
        for i in range(256):
            if i not in self._byte_encoder:
                # Map to unicode characters starting at 256
                self._byte_encoder[i] = chr(256 + n)
                self._byte_decoder[chr(256 + n)] = i
                n += 1

    @staticmethod
    fn from_tiktoken(path: String) raises -> BPETokenizer:
        """
        Load a BPE tokenizer from tiktoken format.

        Args:
            path: Path to the tiktoken vocabulary file.

        Returns:
            A configured BPETokenizer.

        Raises:
            Error if the file cannot be read or parsed.

        Example:
            var tokenizer = BPETokenizer.from_tiktoken("cl100k_base.tiktoken")
        """
        var tokenizer = BPETokenizer()
        # TODO: Implement tiktoken format loading
        # Format: base64-encoded token + space + rank (one per line)
        raise Error("tiktoken loading not yet implemented")

    @staticmethod
    fn from_huggingface(path: String) raises -> BPETokenizer:
        """
        Load a BPE tokenizer from HuggingFace tokenizer.json format.

        Args:
            path: Path to the tokenizer.json file.

        Returns:
            A configured BPETokenizer.

        Raises:
            Error if the file cannot be read or parsed.

        Example:
            var tokenizer = BPETokenizer.from_huggingface("tokenizer.json")
        """
        var tokenizer = BPETokenizer()
        # TODO: Implement HuggingFace JSON format loading
        raise Error("HuggingFace loading not yet implemented")

    fn encode(self, text: String) raises -> List[Int]:
        """
        Encode text into BPE token IDs.

        The encoding process:
        1. Check for special tokens and handle separately
        2. Convert remaining text to bytes
        3. Map bytes to unicode characters
        4. Apply BPE merges iteratively
        5. Look up final tokens in vocabulary

        Args:
            text: The input text to tokenize.

        Returns:
            A list of integer token IDs.
        """
        var result = List[Int]()

        if len(text) == 0:
            return result

        # Handle special tokens first
        var segments = self.special_tokens.split_on_special(text)

        for segment in segments:
            if segment[].is_special:
                # Look up special token directly
                var token_id = self.special_tokens.get_id(segment[].text)
                if token_id >= 0:
                    result.append(token_id)
            else:
                # Encode regular text with BPE
                var token_ids = self._encode_ordinary(segment[].text)
                for tid in token_ids:
                    result.append(tid[])

        return result

    fn _encode_ordinary(self, text: String) raises -> List[Int]:
        """Encode ordinary (non-special) text using BPE."""
        var result = List[Int]()

        # Convert to bytes and then to BPE unicode representation
        var byte_text = text.as_bytes()
        var unicode_text = String()

        for b in byte_text:
            if int(b[]) in self._byte_encoder:
                unicode_text += self._byte_encoder[int(b[])]

        # Get initial tokens (each unicode character)
        var tokens = List[String]()
        for i in range(len(unicode_text)):
            tokens.append(unicode_text[i])

        # Apply BPE merges
        while len(tokens) > 1:
            # Find the highest priority merge
            var best_pair = (-1, -1)
            var best_rank = -1

            for i in range(len(tokens) - 1):
                var pair = tokens[i] + tokens[i + 1]
                var rank = self.vocab.get_merge_rank(pair)
                if rank >= 0 and (best_rank < 0 or rank < best_rank):
                    best_rank = rank
                    best_pair = (i, i + 1)

            if best_rank < 0:
                break  # No more merges possible

            # Apply the merge
            var new_tokens = List[String]()
            var i = 0
            while i < len(tokens):
                if i == best_pair[0]:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        # Convert tokens to IDs
        for token in tokens:
            var token_id = self.vocab.get_id(token[])
            if token_id >= 0:
                result.append(token_id)
            else:
                # Unknown token - encode as individual bytes
                for c in token[]:
                    var byte_id = self.vocab.get_id(String(c))
                    if byte_id >= 0:
                        result.append(byte_id)

        return result

    fn decode(self, tokens: List[Int]) raises -> String:
        """
        Decode token IDs back into text.

        Args:
            tokens: The list of token IDs to decode.

        Returns:
            The reconstructed text string.
        """
        var parts = List[String]()

        for token_id in tokens:
            var token_text = self.vocab.get_text(token_id[])
            if len(token_text) > 0:
                parts.append(token_text)
            else:
                # Check special tokens
                var special_text = self.special_tokens.get_text(token_id[])
                if len(special_text) > 0:
                    parts.append(special_text)

        # Join and convert from BPE unicode back to bytes
        var unicode_text = String()
        for part in parts:
            unicode_text += part[]

        # Convert back to bytes
        var result_bytes = List[UInt8]()
        for i in range(len(unicode_text)):
            var c = unicode_text[i]
            if c in self._byte_decoder:
                result_bytes.append(UInt8(self._byte_decoder[c]))

        return String(result_bytes)

    fn encode_batch(self, texts: List[String]) raises -> List[List[Int]]:
        """
        Encode multiple texts in batch.

        Args:
            texts: List of input texts to tokenize.

        Returns:
            List of token ID lists, one per input text.
        """
        var results = List[List[Int]]()
        for text in texts:
            results.append(self.encode(text[]))
        return results

    fn decode_batch(self, token_lists: List[List[Int]]) raises -> List[String]:
        """
        Decode multiple token lists in batch.

        Args:
            token_lists: List of token ID lists to decode.

        Returns:
            List of decoded strings.
        """
        var results = List[String]()
        for tokens in token_lists:
            results.append(self.decode(tokens[]))
        return results

    fn vocab_size(self) -> Int:
        """Return the total vocabulary size including special tokens."""
        return self.vocab.size() + self.special_tokens.size()

    fn add_special_token(mut self, text: String, id: Int):
        """
        Add a special token to the tokenizer.

        Args:
            text: The special token text (e.g., "<|endoftext|>").
            id: The token ID to assign.
        """
        self.special_tokens.add(text, id)
