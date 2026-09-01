"""BPE encoding algorithms and their state.

This module owns text-to-token-ID conversion.  It deliberately does not load
vocabularies, render chat prompts, add model tensors, or decode token IDs.
"""

from .backtracking import CpuBacktrackBatch
from .byte_trie import ByteTrie, INLINE_TOKEN_BYTE_CAPACITY
from .cache import TokenCache, MergeCache
from .spans import TokenSpan
from .vocabulary import Vocabulary
from ..preprocessing.whitespace import is_boundary_byte, create_boundary_mask
from std.collections.interval import Interval


comptime SIMD_WIDTH: Int = 16
comptime INITIAL_BACKTRACK_SCRATCH_CAPACITY: Int = 256


struct _TextPiece(Copyable, Movable):
    var text: String
    var span: Interval[Int]

    def __init__(out self, text: String, span: Interval[Int]):
        self.text = text
        self.span = span


struct _MergePiece(Copyable, Movable):
    var text: String
    var span: Interval[Int]

    def __init__(out self, text: String, span: Interval[Int]):
        self.text = text
        self.span = span


struct BpeEncoder(Movable):
    """Stateful BPE encoder with explicit, optional optimizations."""

    var vocabulary: Vocabulary
    var _byte_encoder: List[String]
    var _cache: TokenCache
    var _merge_cache: MergeCache
    var _vocabulary_trie: ByteTrie
    var _cache_enabled: Bool
    var _backtracking_enabled: Bool
    var _backtracking_batch: CpuBacktrackBatch[1]

    def __init__(out self, vocabulary: Vocabulary):
        self.vocabulary = vocabulary.copy()
        self._byte_encoder = List[String](capacity=256)
        self._cache = TokenCache(10000)
        self._merge_cache = MergeCache()
        self._vocabulary_trie = ByteTrie()
        self._cache_enabled = True
        self._backtracking_enabled = False
        self._backtracking_batch = CpuBacktrackBatch[1](INITIAL_BACKTRACK_SCRATCH_CAPACITY)
        self._initialize_byte_encoder()
        self._build_vocabulary_trie()

    def encode(mut self, text: String) raises -> List[Int]:
        """Encode ordinary text. Special-token handling is owned by the facade."""
        var spans = self.encode_with_spans(text)
        var result = List[Int]()
        for i in range(len(spans)):
            result.append(spans[i].token_id)
        return result^

    def encode_with_spans(
        mut self,
        text: String,
        byte_offset: Int = 0,
    ) raises -> List[TokenSpan]:
        """Encode ordinary text and retain UTF-8 byte offsets."""
        var result = List[TokenSpan]()
        if text.byte_length() == 0:
            return result^

        var pieces = self._split_at_boundaries(text)
        for piece_index in range(len(pieces)):
            var piece = pieces[piece_index].copy()
            var piece_spans = self._encode_piece_with_spans(piece.text)
            var base = byte_offset + piece.span.start
            for token_index in range(len(piece_spans)):
                result.append(piece_spans[token_index].shifted(base))
        return result^

    def encode_batch(mut self, texts: List[String]) raises -> List[List[Int]]:
        """Encode multiple independent strings in input order."""
        var result = List[List[Int]](capacity=len(texts))
        for i in range(len(texts)):
            result.append(self.encode(texts[i]))
        return result^

    def vocab_size(self) -> Int:
        return self.vocabulary.size()

    def set_cache_enabled(mut self, enabled: Bool):
        self._cache_enabled = enabled

    def cache_enabled(self) -> Bool:
        return self._cache_enabled

    def cache_hit_rate(self) -> Float64:
        return self._cache.hit_rate()

    def cache_stats(self) -> Tuple[Int, Int, Int]:
        return Tuple(self._cache.hits(), self._cache.misses(), self._cache.size())

    def clear_cache(mut self):
        self._cache.clear()

    def set_backtracking_enabled(mut self, enabled: Bool):
        """Enable the retained backtracking algorithm explicitly."""
        if enabled and not self.vocabulary.has_backtrack_tables():
            self.vocabulary.build_backtrack_tables()
        self._backtracking_enabled = enabled

    def backtracking_enabled(self) -> Bool:
        return self._backtracking_enabled

    def reserve_backtracking_capacity(mut self, minimum_capacity: Int):
        """Pre-size the persistent CPU backtracking slot for expected inputs."""
        self._backtracking_batch.reserve(minimum_capacity)

    def backtracking_scratch_capacity(self) -> Int:
        return self._backtracking_batch.slot_capacity()

    def _initialize_byte_encoder(mut self):
        # GPT-2 byte-to-unicode mapping used by the retained BPE implementation.
        for _ in range(256):
            self._byte_encoder.append("")

        var visible = List[Int](capacity=188)
        for value in range(33, 127):
            visible.append(value)
        for value in range(161, 173):
            visible.append(value)
        for value in range(174, 256):
            visible.append(value)

        for i in range(len(visible)):
            self._byte_encoder[visible[i]] = chr(visible[i])

        var next_codepoint = 0
        for value in range(256):
            var is_visible = (value >= 33 and value < 127) or (value >= 161 and value < 173) or (value >= 174)
            if not is_visible:
                self._byte_encoder[value] = chr(256 + next_codepoint)
                next_codepoint += 1

    def _build_vocabulary_trie(mut self):
        for token_id in range(self.vocabulary.size()):
            var token_bytes = InlineArray[UInt8, INLINE_TOKEN_BYTE_CAPACITY](fill=0)
            var token_length = self.vocabulary.copy_token_bytes_to(token_id, token_bytes)
            if token_length > 0:
                self._vocabulary_trie.insert_inline(token_bytes, token_length, token_id)
        self._vocabulary_trie.freeze()

    def _split_at_boundaries(self, text: String) -> List[_TextPiece]:
        var pieces = List[_TextPiece]()
        if text.byte_length() == 0:
            return pieces^

        var start = 0
        var i = 0
        var text_length = text.byte_length()
        var text_pointer = text.unsafe_ptr()

        while i + SIMD_WIDTH <= text_length:
            var chunk = SIMD[DType.uint8, SIMD_WIDTH]()
            comptime for lane in range(SIMD_WIDTH):
                chunk[lane] = text_pointer[unsafe_offset=i + lane]
            var mask = create_boundary_mask(chunk)
            if Int(mask.reduce_add()) > 0:
                comptime for lane in range(SIMD_WIDTH):
                    if mask[lane] == 1:
                        var boundary = i + lane
                        if boundary > start:
                            pieces.append(_TextPiece(String(text[byte=start:boundary]), Interval(start, boundary)))
                        pieces.append(_TextPiece(String(text[byte=boundary]), Interval(boundary, boundary + 1)))
                        start = boundary + 1
            i += SIMD_WIDTH

        while i < text_length:
            if is_boundary_byte(text_pointer[unsafe_offset=i]):
                if i > start:
                    pieces.append(_TextPiece(String(text[byte=start:i]), Interval(start, i)))
                pieces.append(_TextPiece(String(text[byte=i]), Interval(i, i + 1)))
                start = i + 1
            i += 1

        if start < text_length:
            pieces.append(_TextPiece(String(text[byte=start:text_length]), Interval(start, text_length)))
        return pieces^

    def _encode_piece_with_spans(mut self, piece: String) raises -> List[TokenSpan]:
        if self._cache_enabled:
            var cached = self._cache.get(piece)
            if cached:
                return cached.value().copy()

        var token_spans: List[TokenSpan]
        if self._backtracking_enabled:
            token_spans = self._backtracking_batch.encode_slot_with_spans(0, self.vocabulary, self._vocabulary_trie, piece)
        else:
            token_spans = self._merge_encode_with_spans(piece)

        if self._cache_enabled:
            self._cache.put(piece, token_spans.copy())
        return token_spans^

    def _merge_encode_with_spans(mut self, piece: String) raises -> List[TokenSpan]:
        var result = List[TokenSpan]()
        var piece_bytes = piece.as_bytes()
        var tokens = List[_MergePiece](capacity=len(piece_bytes))

        for i in range(len(piece_bytes)):
            var byte_token = self._byte_encoder[Int(piece_bytes[i])]
            tokens.append(_MergePiece(byte_token, Interval(i, i + 1)))

        var buffer = List[_MergePiece](capacity=len(tokens))
        while len(tokens) > 1:
            var best_index = -1
            var best_rank = -1
            for i in range(len(tokens) - 1):
                var first = tokens[i].copy()
                var second = tokens[i + 1].copy()
                var rank = self._merge_cache.get_rank(first.text, second.text)
                if rank < 0:
                    rank = self.vocabulary.get_merge_rank(first.text + second.text)
                if rank >= 0 and (best_rank < 0 or rank < best_rank):
                    best_index = i
                    best_rank = rank

            if best_index < 0:
                break

            buffer.clear()
            var i = 0
            while i < len(tokens):
                if i == best_index:
                    var left = tokens[i].copy()
                    var right = tokens[i + 1].copy()
                    buffer.append(
                        _MergePiece(
                            left.text + right.text,
                            Interval(left.span.start, right.span.end),
                        )
                    )
                    i += 2
                else:
                    buffer.append(tokens[i].copy())
                    i += 1
            var old_tokens = tokens^
            tokens = buffer^
            buffer = old_tokens^

        for i in range(len(tokens)):
            var token = tokens[i].copy()
            var token_id = self.vocabulary.get_id(token.text)
            if token_id >= 0:
                result.append(TokenSpan(token_id, token.span))
            else:
                for j in range(token.text.count_codepoints()):
                    var byte_id = self.vocabulary.get_id(String(token.text[codepoint=j]))
                    if byte_id >= 0:
                        result.append(TokenSpan(byte_id, Interval(token.span.start + j, token.span.start + j + 1)))
        return result^
