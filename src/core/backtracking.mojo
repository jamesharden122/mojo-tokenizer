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
from .spans import TokenSpan
from .vocabulary import Vocabulary
from .byte_trie import ByteTrie, TrieLookupResult
from std.collections.interval import Interval
from std.memory import Pointer
from std.memory.alloc import Allocation, Layout, alloc, dealloc


struct BacktrackScratch(Movable):
    """Reusable CPU buffers for one backtracking encoder slot."""

    var _text: Allocation[UInt8]
    var _tokens: Allocation[Int]
    var _token_spans: Allocation[Interval[Int]]
    var _bitfield: BitField
    var capacity: Int
    var _text_len: Int
    var _token_count: Int

    def __init__(out self, capacity: Int):
        assert capacity > 0, "BacktrackScratch capacity must be positive"
        self._text = alloc(Layout[UInt8](count=capacity))
        self._tokens = alloc(Layout[Int](count=capacity))
        self._token_spans = alloc(Layout[Interval[Int]](count=capacity))
        self._bitfield = BitField(capacity + 1)
        self.capacity = capacity
        self._text_len = 0
        self._token_count = 0

    def __deinit__(deinit self):
        dealloc(self._text^)
        dealloc(self._tokens^)
        dealloc(self._token_spans^)

    def reserve(mut self, minimum_capacity: Int):
        """Grow the reusable buffers when a larger input arrives."""
        if minimum_capacity <= self.capacity:
            return

        var next_text = alloc(Layout[UInt8](count=minimum_capacity))
        var next_tokens = alloc(Layout[Int](count=minimum_capacity))
        var next_token_spans = alloc(Layout[Interval[Int]](count=minimum_capacity))
        var next_bitfield = BitField(minimum_capacity + 1)
        var old_text = self._text^
        var old_tokens = self._tokens^
        var old_token_spans = self._token_spans^
        self._text = next_text^
        self._tokens = next_tokens^
        self._token_spans = next_token_spans^
        self._bitfield = next_bitfield^
        self.capacity = minimum_capacity
        self._text_len = 0
        self._token_count = 0
        dealloc(old_text^)
        dealloc(old_tokens^)
        dealloc(old_token_spans^)

    def reset_text(mut self, text: String):
        var text_len = text.byte_length()
        self.reserve(text_len)
        self._reset(text_len)
        var source = text.unsafe_ptr()
        var storage = self._text.unsafe_span()
        for i in range(self._text_len):
            storage[i] = source[unsafe_offset=i]

    def reset_bytes(mut self, text: List[UInt8]):
        self.reserve(len(text))
        self._reset(len(text))
        var storage = self._text.unsafe_span()
        for i in range(self._text_len):
            storage[i] = text[i]

    def text_length(self) -> Int:
        return self._text_len

    def token_count(self) -> Int:
        return self._token_count

    def _reset(mut self, text_len: Int):
        self._text_len = text_len
        self._token_count = 0
        self._bitfield.reset_all()


struct BacktrackEncoder[
    vocab_origin: ImmOrigin,
    trie_origin: ImmOrigin,
    scratch_origin: MutOrigin,
](Movable):
    """
    O(n) backtracking BPE encoder.

    Uses a BitField to track reachable positions and validates
    token pairs to ensure correct BPE tokenization.

    Usage:
        var scratch = BacktrackScratch(max(1, len(text_bytes)))
        var encoder = BacktrackEncoder(vocab, trie, scratch, text_bytes)
        while encoder.step():
            pass
        var tokens = encoder.get_tokens()
    """

    var _vocab: Pointer[Vocabulary, Self.vocab_origin]
    var _trie: Pointer[ByteTrie, Self.trie_origin]
    var _scratch: Pointer[BacktrackScratch, Self.scratch_origin]
    var _next_token: Int
    var _next_token_len: Int
    var _pos: Int
    var _skip_pair_validation: Bool

    def __init__(
        out self,
        ref[Self.vocab_origin] vocab: Vocabulary,
        ref[Self.trie_origin] trie: ByteTrie,
        ref[Self.scratch_origin] scratch: BacktrackScratch,
        text: String,
        skip_pair_val: Bool = True,
    ):
        self._vocab = Pointer(to=vocab)
        self._trie = Pointer(to=trie)
        self._scratch = Pointer(to=scratch)
        self._next_token = -1
        self._next_token_len = 0
        self._pos = 0
        self._skip_pair_validation = skip_pair_val
        self._scratch[].reset_text(text)
        self._initialize(skip_pair_val)

    def __init__(
        out self,
        ref[Self.vocab_origin] vocab: Vocabulary,
        ref[Self.trie_origin] trie: ByteTrie,
        ref[Self.scratch_origin] scratch: BacktrackScratch,
        text: List[UInt8],
        skip_pair_val: Bool = True,
    ):
        self._vocab = Pointer(to=vocab)
        self._trie = Pointer(to=trie)
        self._scratch = Pointer(to=scratch)
        self._next_token = -1
        self._next_token_len = 0
        self._pos = 0
        self._skip_pair_validation = skip_pair_val
        self._scratch[].reset_bytes(text)
        self._initialize(skip_pair_val)

    def _initialize(mut self, skip_pair_val: Bool):
        self._pos = 0
        self._next_token = -1
        self._next_token_len = 0
        self._skip_pair_validation = skip_pair_val
        if self._scratch[].text_length() > 0:
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
        if pos >= self._scratch[].text_length():
            return TrieLookupResult(False, -1, 0)

        # Look up directly in the owned input buffer; avoid allocating a
        # suffix List for every match attempt.
        var text = self._scratch[]._text.unsafe_span()
        var node_idx = 0
        var last_match_id = -1
        var last_match_len = 0
        for i in range(pos, self._scratch[].text_length()):
            var edge_index = self._trie[]._child_edge_index(node_idx, text[i])
            if edge_index < 0:
                break
            node_idx = self._trie[].edges[edge_index].child
            if self._trie[]._is_terminal(node_idx):
                last_match_id = self._trie[]._token_id_for_terminal_edge(edge_index)
                last_match_len = i - pos + 1
        return TrieLookupResult(last_match_id >= 0, last_match_id, last_match_len)

    def _get_token_len(self, token_id: Int) -> Int:
        """
        Get the byte length of a token.

        Args:
            token_id: Token ID.

        Returns:
            Number of bytes in the token.
        """
        return self._vocab[].get_byte_length(token_id)

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
        if self._scratch[].token_count() > 0:
            last_token = self._scratch[]._tokens.unsafe_span()[self._scratch[].token_count() - 1]

        while True:
            var end_pos = self._pos + token_len

            # Check if:
            # 1. End position is reachable (bit is set)
            # 2. Token pair is valid (if we have a previous token AND validation is enabled)
            var is_reachable = self._scratch[]._bitfield.is_set(end_pos)
            var is_valid_pair = True
            if last_token >= 0 and not self._skip_pair_validation:
                is_valid_pair = self._vocab[].is_valid_token_pair(last_token, token)

            if is_reachable and is_valid_pair:
                # Accept this token
                var token_index = self._scratch[].token_count()
                self._scratch[]._tokens.unsafe_span()[token_index] = token
                self._scratch[]._token_spans.unsafe_span()[token_index] = Interval(self._pos, end_pos)
                self._scratch[]._token_count += 1
                self._pos = end_pos

                # Find next longest match
                var result = self._next_match(end_pos)
                self._next_token = result.token_id
                self._next_token_len = result.match_length
                break
            else:
                # Try shorter prefix token
                var shorter = self._vocab[].get_next_prefix(token)
                if shorter >= 0:
                    token = shorter
                    token_len = self._get_token_len(token)
                else:
                    # No shorter prefix - must backtrack
                    # Mark current position as unreachable
                    self._scratch[]._bitfield.clear(self._pos)

                    # Pop the last token
                    if self._scratch[].token_count() > 0:
                        self._scratch[]._token_count -= 1
                        var popped_index = self._scratch[].token_count()
                        var popped = self._scratch[]._tokens.unsafe_span()[popped_index]
                        var popped_span = self._scratch[]._token_spans.unsafe_span()[popped_index].copy()
                        self._pos = popped_span.start

                        # The popped token becomes our next token to try
                        self._next_token = popped
                        self._next_token_len = popped_span.end - popped_span.start
                    else:
                        # No tokens to pop - encoding failed
                        self._next_token = -1
                        self._next_token_len = 0
                    break

        return self._next_token >= 0 or self._pos < self._scratch[].text_length()

    def encode(mut self) -> List[Int]:
        """
        Encode the entire input text.

        Returns:
            List of token IDs.
        """
        while self.step():
            pass
        return self.get_tokens()

    def encode_with_spans(mut self) -> List[TokenSpan]:
        """Encode the input and retain each token's relative byte span."""
        while self.step():
            pass
        return self.get_token_spans()

    def count(self) -> Int:
        """Get current token count."""
        return self._scratch[].token_count()

    def pos(self) -> Int:
        """Get current position in text."""
        return self._pos

    def get_tokens(self) -> List[Int]:
        """Get the accumulated tokens (copy)."""
        var result = List[Int](capacity=self._scratch[].token_count())
        var tokens = self._scratch[]._tokens.unsafe_span()
        for i in range(self._scratch[].token_count()):
            result.append(tokens[i])
        return result^

    def get_token_spans(self) -> List[TokenSpan]:
        """Get accumulated IDs and byte spans as a copied API result."""
        var result = List[TokenSpan](capacity=self._scratch[].token_count())
        var tokens = self._scratch[]._tokens.unsafe_span()
        var spans = self._scratch[]._token_spans.unsafe_span()
        for i in range(self._scratch[].token_count()):
            result.append(TokenSpan(tokens[i], spans[i]))
        return result^

    def into_tokens(mut self) -> List[Int]:
        """Return the current token IDs as a List at the API boundary."""
        return self.get_tokens()


struct CpuBacktrackBatch[BatchSize: Int](Movable):
    """Fixed CPU batch slots with reusable allocation-backed backtracking state."""

    var slots: InlineArray[BacktrackScratch, Self.BatchSize]
    var capacity: Int

    def __init__(out self, capacity: Int):
        assert Self.BatchSize > 0, "CpuBacktrackBatch requires at least one slot"
        assert capacity > 0, "CpuBacktrackBatch capacity must be positive"
        self.slots = InlineArray[BacktrackScratch, Self.BatchSize](uninitialized=True)
        var slots = self.slots.unsafe_ptr()
        comptime for i in range(Self.BatchSize):
            slots.unsafe_offset(i).unsafe_write(BacktrackScratch(capacity))
        self.capacity = capacity

    def slot_count(self) -> Int:
        return Self.BatchSize

    def slot_capacity(self) -> Int:
        return self.capacity

    def reserve(mut self, minimum_capacity: Int):
        """Grow every slot together so their batch capacity stays uniform."""
        if minimum_capacity <= self.capacity:
            return

        var next_capacity = max(minimum_capacity, self.capacity * 2)
        for slot in range(Self.BatchSize):
            self.slots[slot].reserve(next_capacity)
        self.capacity = next_capacity

    def encode_slot(
        mut self,
        slot: Int,
        vocab: Vocabulary,
        trie: ByteTrie,
        text: String,
        skip_pair_val: Bool = True,
    ) raises -> List[Int]:
        self._check_slot(slot)
        self.reserve(text.byte_length())
        var encoder = BacktrackEncoder(vocab, trie, self.slots[slot], text, skip_pair_val)
        return encoder.encode()

    def encode_slot_bytes(
        mut self,
        slot: Int,
        vocab: Vocabulary,
        trie: ByteTrie,
        text: List[UInt8],
        skip_pair_val: Bool = True,
    ) raises -> List[Int]:
        self._check_slot(slot)
        self.reserve(len(text))
        var encoder = BacktrackEncoder(vocab, trie, self.slots[slot], text, skip_pair_val)
        return encoder.encode()

    def encode_slot_with_spans(
        mut self,
        slot: Int,
        vocab: Vocabulary,
        trie: ByteTrie,
        text: String,
        skip_pair_val: Bool = True,
    ) raises -> List[TokenSpan]:
        self._check_slot(slot)
        self.reserve(text.byte_length())
        var encoder = BacktrackEncoder(vocab, trie, self.slots[slot], text, skip_pair_val)
        return encoder.encode_with_spans()

    def _check_slot(self, slot: Int) raises:
        if slot < 0 or slot >= Self.BatchSize:
            raise Error("CpuBacktrackBatch slot index out of bounds")


def backtrack_encode_into(
    mut scratch: BacktrackScratch,
    vocab: Vocabulary,
    trie: ByteTrie,
    text: String,
    skip_pair_val: Bool = True,
) -> List[Int]:
    """Encode into caller-owned reusable CPU scratch storage."""
    var encoder = BacktrackEncoder(vocab, trie, scratch, text, skip_pair_val)
    return encoder.encode()


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
    var scratch = BacktrackScratch(max(1, text.byte_length()))
    return backtrack_encode_into(scratch, vocab, trie, text)


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
    var scratch = BacktrackScratch(max(1, len(text_bytes)))
    var encoder = BacktrackEncoder(vocab, trie, scratch, text_bytes)
    return encoder.encode()
