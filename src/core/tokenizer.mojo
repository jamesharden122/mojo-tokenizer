"""Public encoding facade.

`BpeTokenizer` coordinates special-token recognition and ordinary BPE
encoding. Loading, decoding, model-buffer construction, and prompt formatting
are intentionally separate responsibilities.
"""

from .encoder import BpeEncoder
from .byte_patterns import BytePatternSet
from .byte_trie import ByteTrie
from .special_tokens import SpecialTokenSet
from .spans import TokenSpan
from .vocabulary import Vocabulary
from std.collections.interval import Interval


trait Tokenizer:
    def encode(mut self, text: String) raises -> List[Int]:
        ...

    def encode_with_spans(mut self, text: String) raises -> List[TokenSpan]:
        ...

    def vocab_size(self) -> Int:
        ...


struct BpeTokenizer(Movable):
    var _encoder: BpeEncoder
    var _special_tokens: SpecialTokenSet
    var _pattern_trie: ByteTrie
    var _pattern_count: Int

    def __init__(out self, vocabulary: Vocabulary, special_tokens: SpecialTokenSet):
        self._encoder = BpeEncoder(vocabulary)
        self._special_tokens = special_tokens.copy()
        self._pattern_trie = ByteTrie()
        self._pattern_count = 0
        self._pattern_trie.freeze()

    def __init__(
        out self,
        vocabulary: Vocabulary,
        special_tokens: SpecialTokenSet,
        byte_patterns: BytePatternSet,
    ) raises:
        self._encoder = BpeEncoder(vocabulary)
        self._special_tokens = special_tokens.copy()
        self._pattern_trie = ByteTrie()
        self._pattern_count = byte_patterns.size()
        for i in range(byte_patterns.size()):
            var pattern = byte_patterns.get(i)
            if not vocabulary.has_id(pattern.token_id):
                raise Error("Byte pattern token ID must exist in Vocabulary")
            self._pattern_trie.insert_intervals(pattern.intervals, pattern.token_id)
        self._pattern_trie.freeze()

    def encode(mut self, text: String) raises -> List[Int]:
        var spans = self.encode_with_spans(text)
        var result = List[Int]()
        for i in range(len(spans)):
            result.append(spans[i].token_id)
        return result^

    def encode_with_spans(mut self, text: String) raises -> List[TokenSpan]:
        var result = List[TokenSpan]()
        if text.byte_length() == 0:
            return result^

        var segments = self._special_tokens.split_on_special(text)
        for i in range(len(segments)):
            var segment = segments[i].copy()
            if segment.is_special:
                var token_id = self._special_tokens.get_id(segment.text)
                if token_id >= 0:
                    result.append(TokenSpan(token_id, segment.span))
            else:
                var ordinary_spans = self._encode_ordinary_with_patterns(segment.text, segment.span.start)
                for j in range(len(ordinary_spans)):
                    result.append(ordinary_spans[j].copy())
        return result^

    def encode_batch(mut self, texts: List[String]) raises -> List[List[Int]]:
        var result = List[List[Int]](capacity=len(texts))
        for i in range(len(texts)):
            result.append(self.encode(texts[i]))
        return result^

    def encode_batch_with_spans(mut self, texts: List[String]) raises -> List[List[TokenSpan]]:
        """Encode multiple strings with byte spans relative to each input."""
        var result = List[List[TokenSpan]](capacity=len(texts))
        for i in range(len(texts)):
            result.append(self.encode_with_spans(texts[i]))
        return result^

    def vocab_size(self) -> Int:
        return self._encoder.vocab_size() + self._special_tokens.size()

    def set_cache_enabled(mut self, enabled: Bool):
        self._encoder.set_cache_enabled(enabled)

    def cache_hit_rate(self) -> Float64:
        return self._encoder.cache_hit_rate()

    def clear_cache(mut self):
        self._encoder.clear_cache()

    def set_backtracking_enabled(mut self, enabled: Bool):
        self._encoder.set_backtracking_enabled(enabled)

    def reserve_backtracking_capacity(mut self, minimum_capacity: Int):
        """Pre-size the reusable CPU backtracking slot for expected inputs."""
        self._encoder.reserve_backtracking_capacity(minimum_capacity)

    def backtracking_scratch_capacity(self) -> Int:
        return self._encoder.backtracking_scratch_capacity()

    def _encode_ordinary_with_patterns(
        mut self,
        text: String,
        byte_offset: Int,
    ) raises -> List[TokenSpan]:
        if self._pattern_count == 0:
            return self._encoder.encode_with_spans(text, byte_offset)

        var result = List[TokenSpan]()
        var ordinary_start = 0
        var position = 0
        while position < text.byte_length():
            if not self._is_utf8_boundary(text, position):
                position += 1
                continue
            var lookup = self._pattern_trie.lookup_string_at_offset(text, position)
            if not lookup.found or not self._is_utf8_boundary(text, position + lookup.match_length):
                position += 1
                continue

            if position > ordinary_start:
                var ordinary = String(text[byte=ordinary_start:position])
                var ordinary_spans = self._encoder.encode_with_spans(ordinary, byte_offset + ordinary_start)
                for i in range(len(ordinary_spans)):
                    result.append(ordinary_spans[i].copy())

            result.append(
                TokenSpan(
                    lookup.token_id,
                    Interval(byte_offset + position, byte_offset + position + lookup.match_length),
                )
            )
            position += lookup.match_length
            ordinary_start = position

        if ordinary_start < text.byte_length():
            var ordinary = String(text[byte = ordinary_start : text.byte_length()])
            var ordinary_spans = self._encoder.encode_with_spans(ordinary, byte_offset + ordinary_start)
            for i in range(len(ordinary_spans)):
                result.append(ordinary_spans[i].copy())
        return result^

    def _is_utf8_boundary(self, text: String, byte_offset: Int) -> Bool:
        if byte_offset <= 0 or byte_offset >= text.byte_length():
            return True
        var byte = text.unsafe_ptr()[unsafe_offset=byte_offset]
        return byte < UInt8(128) or byte >= UInt8(192)
