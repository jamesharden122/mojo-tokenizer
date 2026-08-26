"""Public encoding facade.

`BpeTokenizer` coordinates special-token recognition and ordinary BPE
encoding. Loading, decoding, model-buffer construction, and prompt formatting
are intentionally separate responsibilities.
"""

from .encoder import BpeEncoder
from .special_tokens import SpecialTokenSet
from .vocabulary import Vocabulary


trait Tokenizer:
    def encode(mut self, text: String) raises -> List[Int]:
        ...

    def vocab_size(self) -> Int:
        ...


struct BpeTokenizer(Movable):
    var _encoder: BpeEncoder
    var _special_tokens: SpecialTokenSet

    def __init__(
        out self, vocabulary: Vocabulary, special_tokens: SpecialTokenSet
    ):
        self._encoder = BpeEncoder(vocabulary)
        self._special_tokens = special_tokens.copy()

    def encode(mut self, text: String) raises -> List[Int]:
        var result = List[Int]()
        if text.byte_length() == 0:
            return result^

        var segments = self._special_tokens.split_on_special(text)
        for i in range(len(segments)):
            var segment = segments[i].copy()
            if segment.is_special:
                var token_id = self._special_tokens.get_id(segment.text)
                if token_id >= 0:
                    result.append(token_id)
            else:
                var ordinary_ids = self._encoder.encode(segment.text)
                for j in range(len(ordinary_ids)):
                    result.append(ordinary_ids[j])
        return result^

    def encode_batch(mut self, texts: List[String]) raises -> List[List[Int]]:
        var result = List[List[Int]](capacity=len(texts))
        for i in range(len(texts)):
            result.append(self.encode(texts[i]))
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
