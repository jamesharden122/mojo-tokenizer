"""
O(n) Backtracking BPE Encoder.

This implements rs-bpe's backtracking algorithm for O(n) BPE encoding.
It's a lazy variation of dynamic programming that only computes states
that must be visited for a given input.

Key insight: Instead of computing all possible tokenizations (O(n²)),
we greedily try longest matches and backtrack when validation fails.

Algorithm:
1. Find longest token match at position (via trie)
2. Check if (end_pos reachable) AND (token pair valid)
3. If valid: accept token, advance position
4. If not: try shorter prefix, or backtrack
5. Repeat until end of input

Performance: O(n) amortized - each position visited at most twice
(once when advancing, once when backtracking).

Ported from rs-bpe's backtrack_encoder.rs implementation.

PAIR VALIDATION:
By default (skip_pair_validation=True), pair validation is disabled. This
makes the encoder use greedy longest-match behavior. Callers can enable pair
validation for vocabularies whose merge metadata requires it.

For custom BPE vocabularies where every token encodes to itself,
set skip_pair_validation=False to enable full backtracking with
pair validation.
"""

from .bitfield import BitField
from .vocabulary import Vocabulary
from .byte_trie import ByteTrie, TrieLookupResult


struct BacktrackEncoder(Movable):
    """
    O(n) backtracking BPE encoder.

    Uses a BitField to track reachable positions and validates
    token pairs to ensure correct BPE tokenization.

    Usage:
        var encoder = BacktrackEncoder(vocab, trie, text_bytes)
        while encoder.step():
            pass
        var tokens = encoder.get_tokens()
    """

    var _vocab: Vocabulary
    """Vocabulary with backtracking tables."""

    var _trie: ByteTrie
    """Byte trie for longest match lookups."""

    var _text: List[UInt8]
    """Input text as bytes."""

    var _tokens: List[Int]
    """Accumulated token IDs."""

    var _next_token: Int
    """Next token to try, or -1 if none."""

    var _next_token_len: Int
    """Length of next token in bytes."""

    var _pos: Int
    """Current position in text."""

    var _bitfield: BitField
    """Tracks reachable positions (all start as reachable)."""

    var _skip_pair_validation: Bool
    """If True, skip vocabulary pair-validation checks."""

    def __init__(
        out self,
        var vocab: Vocabulary,
        var trie: ByteTrie,
        text: List[UInt8],
        skip_pair_validation: Bool = True,
    ):
        """
        Create a new backtracking encoder.

        Args:
            vocab: Vocabulary with backtracking tables built.
            trie: Byte trie for token lookups.
            text: Input text as bytes.
            skip_pair_validation: If True, skip is_valid_token_pair checks.
                Defaults to True for greedy longest-match behavior.
        """
        self._vocab = vocab^
        self._trie = trie^
        self._text = text.copy()
        self._tokens = List[Int](capacity=text.byte_length() // 3)
        self._pos = 0
        # Initialize these first to satisfy Mojo's initialization requirements
        self._next_token = -1
        self._next_token_len = 0
        # All positions are initially reachable (bits set to 1)
        self._bitfield = BitField(text.byte_length() + 1)
        self._skip_pair_validation = skip_pair_validation

        # Find first longest match
        if len(self._text) > 0:
            var result = self._next_match(0)
            self._next_token = result.token_id
            self._next_token_len = result.match_length

    def _next_match(self, pos: Int) -> TrieLookupResult:
        """
        Find the longest token match starting at position.

        Args:
            pos: Starting position in text.

        Returns:
            TrieLookupResult with token_id and match_length.
        """
        if pos >= len(self._text):
            return TrieLookupResult(False, -1, 0)

        # Create a view of remaining text
        var remaining = List[UInt8](capacity=len(self._text) - pos)
        for i in range(pos, len(self._text)):
            remaining.append(self._text[i])

        return self._trie.lookup(remaining)

    def _get_token_len(self, token_id: Int) -> Int:
        """
        Get the byte length of a token.

        Args:
            token_id: Token ID.

        Returns:
            Number of bytes in the token.
        """
        # Get token text and return its byte length
        var token_text = self._vocab.get_text(token_id)
        return len(token_text.as_bytes())

    def step(mut self) -> Bool:
        """
        Perform one step of the backtracking algorithm.

        Returns:
            True if more tokens remain to process, False when done.
        """
        if self._next_token < 0:
            return False

        var token = self._next_token
        var token_len = self._next_token_len

        # Get last token for pair validation
        var last_token = -1
        if len(self._tokens) > 0:
            last_token = self._tokens[len(self._tokens) - 1]

        while True:
            var end_pos = self._pos + token_len

            # Check if:
            # 1. End position is reachable (bit is set)
            # 2. Token pair is valid (if we have a previous token AND validation is enabled)
            var is_reachable = self._bitfield.is_set(end_pos)
            var is_valid_pair = True
            if last_token >= 0 and not self._skip_pair_validation:
                is_valid_pair = self._vocab.is_valid_token_pair(last_token, token)

            if is_reachable and is_valid_pair:
                # Accept this token
                self._tokens.append(token)
                self._pos = end_pos

                # Find next longest match
                var result = self._next_match(end_pos)
                self._next_token = result.token_id
                self._next_token_len = result.match_length
                break
            else:
                # Try shorter prefix token
                var shorter = self._vocab.get_next_prefix(token)
                if shorter >= 0:
                    token = shorter
                    token_len = self._get_token_len(token)
                else:
                    # No shorter prefix - must backtrack
                    # Mark current position as unreachable
                    self._bitfield.clear(self._pos)

                    # Pop the last token
                    if len(self._tokens) > 0:
                        var popped = self._tokens.pop()
                        var popped_len = self._get_token_len(popped)
                        self._pos -= popped_len

                        # The popped token becomes our next token to try
                        self._next_token = popped
                        self._next_token_len = popped_len
                    else:
                        # No tokens to pop - encoding failed
                        self._next_token = -1
                        self._next_token_len = 0
                    break

        return self._next_token >= 0 or self._pos < len(self._text)

    def encode(mut self) -> List[Int]:
        """
        Encode the entire input text.

        Returns:
            List of token IDs.
        """
        while self.step():
            pass
        return self._tokens.copy()

    def count(self) -> Int:
        """Get current token count."""
        return len(self._tokens)

    def pos(self) -> Int:
        """Get current position in text."""
        return self._pos

    def get_tokens(self) -> List[Int]:
        """Get the accumulated tokens (copy)."""
        return self._tokens.copy()

    def into_tokens(deinit self) -> List[Int]:
        """Consume encoder and return tokens."""
        return self._tokens^


def backtrack_encode(
    vocab: Vocabulary,
    trie: ByteTrie,
    text: String,
) -> List[Int]:
    """
    Encode text using the O(n) backtracking algorithm.

    Args:
        vocab: Vocabulary with backtracking tables.
        trie: Byte trie for token lookups.
        text: Input text string.

    Returns:
        List of token IDs.
    """
    # Convert string to bytes
    var text_bytes = text.as_bytes()
    var byte_list = List[UInt8](capacity=len(text_bytes))
    for i in range(len(text_bytes)):
        byte_list.append(text_bytes[i])

    var encoder = BacktrackEncoder(vocab.copy(), trie.copy(), byte_list)
    return encoder.encode()


def backtrack_encode_bytes(
    vocab: Vocabulary,
    trie: ByteTrie,
    text_bytes: List[UInt8],
) -> List[Int]:
    """
    Encode bytes using the O(n) backtracking algorithm.

    Args:
        vocab: Vocabulary with backtracking tables.
        trie: Byte trie for token lookups.
        text_bytes: Input bytes.

    Returns:
        List of token IDs.
    """
    var encoder = BacktrackEncoder(vocab.copy(), trie.copy(), text_bytes)
    return encoder.encode()
