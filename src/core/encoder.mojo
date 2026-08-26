"""BPE encoding algorithms and their state.

This module owns text-to-token-ID conversion.  It deliberately does not load
vocabularies, render chat prompts, add model tensors, or decode token IDs.
"""

from .backtracking import backtrack_encode
from .byte_trie import ByteTrie
from .cache import TokenCache, MergeCache
from .vocabulary import Vocabulary
from ..preprocessing.whitespace import is_boundary_byte, create_boundary_mask


comptime SIMD_WIDTH: Int = 16


struct BpeEncoder(Movable):
    """Stateful BPE encoder with explicit, optional optimizations."""

    var vocabulary: Vocabulary
    var _byte_encoder: List[String]
    var _cache: TokenCache
    var _merge_cache: MergeCache
    var _vocabulary_trie: ByteTrie
    var _cache_enabled: Bool
    var _backtracking_enabled: Bool

    def __init__(out self, vocabulary: Vocabulary):
        self.vocabulary = vocabulary.copy()
        self._byte_encoder = List[String](capacity=256)
        self._cache = TokenCache(10000)
        self._merge_cache = MergeCache()
        self._vocabulary_trie = ByteTrie()
        self._cache_enabled = True
        self._backtracking_enabled = False
        self._initialize_byte_encoder()
        self._build_vocabulary_trie()

    def encode(mut self, text: String) raises -> List[Int]:
        """Encode ordinary text. Special-token handling is owned by the facade."""
        var result = List[Int]()
        if text.byte_length() == 0:
            return result^

        var pieces = self._split_at_boundaries(text)
        for piece_index in range(len(pieces)):
            var piece_ids = self._encode_piece(pieces[piece_index])
            for token_index in range(len(piece_ids)):
                result.append(piece_ids[token_index])
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

    def _initialize_byte_encoder(mut self):
        # GPT-2 byte-to-unicode mapping used by the retained BPE implementation.
        for _ in range(256):
            self._byte_encoder.append("")

        var visible = List[Int]()
        for value in range(33, 127):
            visible.append(value)
        for value in range(161, 173):
            visible.append(value)
        for value in range(174, 256):
            visible.append(value)

        var is_visible = List[Bool](capacity=256)
        for _ in range(256):
            is_visible.append(False)
        for i in range(len(visible)):
            is_visible[visible[i]] = True
            self._byte_encoder[visible[i]] = chr(visible[i])

        var next_codepoint = 0
        for value in range(256):
            if not is_visible[value]:
                self._byte_encoder[value] = chr(256 + next_codepoint)
                next_codepoint += 1

    def _build_vocabulary_trie(mut self):
        for token_id in range(self.vocabulary.size()):
            var token_bytes = self.vocabulary.get_bytes(token_id)
            if len(token_bytes) == 0:
                var token_text = self.vocabulary.get_text(token_id)
                var text_bytes = token_text.as_bytes()
                for i in range(len(text_bytes)):
                    token_bytes.append(text_bytes[i])
            if len(token_bytes) > 0:
                self._vocabulary_trie.insert(token_bytes, token_id)

    def _split_at_boundaries(self, text: String) -> List[String]:
        var pieces = List[String]()
        if text.byte_length() == 0:
            return pieces^

        var start = 0
        var i = 0
        var text_length = text.byte_length()
        var text_pointer = text.unsafe_ptr()

        while i + SIMD_WIDTH <= text_length:
            var chunk = SIMD[DType.uint8, SIMD_WIDTH]()
            comptime
            for lane in range(SIMD_WIDTH):
                chunk[lane] = text_pointer[unsafe_offset=i + lane]
            var mask = create_boundary_mask(chunk)
            if Int(mask.reduce_add()) > 0:
                comptime
                for lane in range(SIMD_WIDTH):
                    if mask[lane] == 1:
                        var boundary = i + lane
                        if boundary > start:
                            pieces.append(String(text[byte=start:boundary]))
                        pieces.append(String(text[byte=boundary]))
                        start = boundary + 1
            i += SIMD_WIDTH

        while i < text_length:
            if is_boundary_byte(text_pointer[unsafe_offset=i]):
                if i > start:
                    pieces.append(String(text[byte=start:i]))
                pieces.append(String(text[byte=i]))
                start = i + 1
            i += 1

        if start < text_length:
            pieces.append(String(text[byte=start:text_length]))
        return pieces^

    def _encode_piece(mut self, piece: String) raises -> List[Int]:
        if self._cache_enabled:
            var cached = self._cache.get(piece)
            if cached:
                return cached.value().copy()

        var token_ids: List[Int]
        if self._backtracking_enabled:
            token_ids = backtrack_encode(
                self.vocabulary.copy(), self._vocabulary_trie.copy(), piece
            )
        else:
            token_ids = self._merge_encode(piece)

        if self._cache_enabled:
            self._cache.put(piece, token_ids.copy())
        return token_ids^

    def _merge_encode(mut self, piece: String) raises -> List[Int]:
        var result = List[Int]()
        var piece_bytes = piece.as_bytes()
        var tokens = List[String](capacity=len(piece_bytes) * 2)

        for i in range(len(piece_bytes)):
            var byte_token = self._byte_encoder[Int(piece_bytes[i])]
            for j in range(byte_token.byte_length()):
                tokens.append(String(byte_token[byte=j]))

        var buffer = List[String](capacity=len(tokens))
        while len(tokens) > 1:
            var best_index = -1
            var best_rank = -1
            for i in range(len(tokens) - 1):
                var rank = self._merge_cache.get_rank(tokens[i], tokens[i + 1])
                if rank < 0:
                    var first = tokens[i]
                    var second = tokens[i + 1]
                    rank = self.vocabulary.get_merge_rank(first + second)
                if rank >= 0 and (best_rank < 0 or rank < best_rank):
                    best_index = i
                    best_rank = rank

            if best_index < 0:
                break

            buffer.clear()
            var i = 0
            while i < len(tokens):
                if i == best_index:
                    var left = tokens[i]
                    var right = tokens[i + 1]
                    buffer.append(left + right)
                    i += 2
                else:
                    buffer.append(tokens[i])
                    i += 1
            var old_tokens = tokens^
            tokens = buffer^
            buffer = old_tokens^

        for i in range(len(tokens)):
            var token_id = self.vocabulary.get_id(tokens[i])
            if token_id >= 0:
                result.append(token_id)
            else:
                for j in range(tokens[i].byte_length()):
                    var byte_id = self.vocabulary.get_id(String(tokens[i][byte=j]))
                    if byte_id >= 0:
                        result.append(byte_id)
        return result^
