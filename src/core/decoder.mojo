"""Optional token-ID decoder for diagnostics and round-trip tests."""

from .special_tokens import SpecialTokenSet
from .vocabulary import Vocabulary


struct TokenDecoder(Copyable, Movable):
    var _vocabulary: Vocabulary
    var _special_tokens: SpecialTokenSet

    def __init__(out self, vocabulary: Vocabulary, special_tokens: SpecialTokenSet):
        self._vocabulary = vocabulary.copy()
        self._special_tokens = special_tokens.copy()

    def decode(self, token_ids: List[Int]) raises -> String:
        var result_bytes = List[UInt8]()
        for i in range(len(token_ids)):
            var token_bytes = self._vocabulary.get_bytes(token_ids[i])
            if len(token_bytes) == 0:
                var token_text = self._vocabulary.get_text(token_ids[i])
                if token_text.byte_length() == 0:
                    token_text = self._special_tokens.get_text(token_ids[i])
                var text_bytes = token_text.as_bytes()
                for j in range(len(text_bytes)):
                    token_bytes.append(text_bytes[j])
            for j in range(len(token_bytes)):
                result_bytes.append(token_bytes[j])

        var result = String()
        var i = 0
        while i < len(result_bytes):
            var first = result_bytes[i]
            if first < 128:
                result += chr(Int(first))
                i += 1
            elif first < 192:
                result += chr(Int(first))
                i += 1
            elif first < 224 and i + 1 < len(result_bytes):
                var code = ((Int(first) & 0x1F) << 6) | (Int(result_bytes[i + 1]) & 0x3F)
                result += chr(code)
                i += 2
            elif first < 240 and i + 2 < len(result_bytes):
                var code = (
                    ((Int(first) & 0x0F) << 12)
                    | ((Int(result_bytes[i + 1]) & 0x3F) << 6)
                    | (Int(result_bytes[i + 2]) & 0x3F)
                )
                result += chr(code)
                i += 3
            elif i + 3 < len(result_bytes):
                var code = (
                    ((Int(first) & 0x07) << 18)
                    | ((Int(result_bytes[i + 1]) & 0x3F) << 12)
                    | ((Int(result_bytes[i + 2]) & 0x3F) << 6)
                    | (Int(result_bytes[i + 3]) & 0x3F)
                )
                result += chr(code)
                i += 4
            else:
                result += chr(Int(first))
                i += 1
        return result
