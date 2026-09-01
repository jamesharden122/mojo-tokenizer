"""Minimal deterministic byte-level BPE trainer for in-memory text chunks."""

from ..core.vocabulary import Vocabulary


comptime BYTE_VOCAB_SIZE: Int = 256


struct BpeTrainer(Movable):
    """Train a frozen-by-convention byte-level BPE vocabulary from text chunks."""

    var target_vocab_size: Int
    var min_pair_frequency: Int
    var _byte_tokens: List[String]

    def __init__(
        out self,
        target_vocab_size: Int = 512,
        min_pair_frequency: Int = 2,
    ):
        assert target_vocab_size >= BYTE_VOCAB_SIZE, "Target vocabulary must include all byte tokens"
        assert min_pair_frequency > 0, "Minimum pair frequency must be positive"
        self.target_vocab_size = target_vocab_size
        self.min_pair_frequency = min_pair_frequency
        self._byte_tokens = _make_byte_tokens()

    def train(self, text_chunks: List[String]) -> Vocabulary:
        """Train byte BPE merges; each input string is an independent sequence."""
        var vocabulary = self._make_base_vocabulary()
        var sequences = List[List[String]](capacity=len(text_chunks))
        for chunk_index in range(len(text_chunks)):
            sequences.append(self._tokenize_bytes(text_chunks[chunk_index]))

        var next_token_id = BYTE_VOCAB_SIZE
        var merge_rank = 0
        while next_token_id < self.target_vocab_size:
            var pair = self._most_frequent_new_pair(sequences, vocabulary)
            if pair[2] < self.min_pair_frequency:
                break

            var first = pair[0]
            var second = pair[1]
            var merged = first + second
            var raw_bytes = self._merged_raw_bytes(vocabulary, first, second)
            vocabulary.add_token_bytes(merged, next_token_id, raw_bytes)
            vocabulary.add_merge(merged, merge_rank)
            self._replace_pair(sequences, first, second, merged)
            next_token_id += 1
            merge_rank += 1

        return vocabulary^

    def _make_base_vocabulary(self) -> Vocabulary:
        var vocabulary = Vocabulary()
        for byte_value in range(BYTE_VOCAB_SIZE):
            var raw_byte = List[UInt8](capacity=1)
            raw_byte.append(UInt8(byte_value))
            vocabulary.add_token_bytes(self._byte_tokens[byte_value], byte_value, raw_byte)
        return vocabulary^

    def _tokenize_bytes(self, text: String) -> List[String]:
        var tokens = List[String](capacity=text.byte_length())
        var source = text.unsafe_ptr()
        for index in range(text.byte_length()):
            tokens.append(self._byte_tokens[Int(source[unsafe_offset=index])])
        return tokens^

    def _most_frequent_new_pair(
        self,
        sequences: List[List[String]],
        vocabulary: Vocabulary,
    ) -> Tuple[String, String, Int]:
        var pair_counts = Dict[String, Int]()
        for sequence_index in range(len(sequences)):
            var sequence = sequences[sequence_index].copy()
            for token_index in range(len(sequence) - 1):
                var pair_key = sequence[token_index] + "\u0000" + sequence[token_index + 1]
                var count = 0
                if pair_key in pair_counts:
                    try:
                        count = pair_counts[pair_key]
                    except:
                        pass
                pair_counts[pair_key] = count + 1

        var best_first = ""
        var best_second = ""
        var best_count = 0
        for sequence_index in range(len(sequences)):
            var sequence = sequences[sequence_index].copy()
            for token_index in range(len(sequence) - 1):
                var first = sequence[token_index]
                var second = sequence[token_index + 1]
                if vocabulary.has_token(first + second):
                    continue
                var pair_key = first + "\u0000" + second
                try:
                    var count = pair_counts[pair_key]
                    if count > best_count:
                        best_first = first
                        best_second = second
                        best_count = count
                except:
                    pass

        return (best_first, best_second, best_count)

    def _merged_raw_bytes(
        self,
        vocabulary: Vocabulary,
        first: String,
        second: String,
    ) -> List[UInt8]:
        var raw_bytes = vocabulary.get_bytes(vocabulary.get_id(first))
        var second_bytes = vocabulary.get_bytes(vocabulary.get_id(second))
        for index in range(len(second_bytes)):
            raw_bytes.append(second_bytes[index])
        return raw_bytes^

    def _replace_pair(
        self,
        mut sequences: List[List[String]],
        first: String,
        second: String,
        merged: String,
    ):
        for sequence_index in range(len(sequences)):
            var source = sequences[sequence_index].copy()
            var replacement = List[String](capacity=len(source))
            var token_index = 0
            while token_index < len(source):
                if token_index + 1 < len(source) and source[token_index] == first and source[token_index + 1] == second:
                    replacement.append(merged)
                    token_index += 2
                else:
                    replacement.append(source[token_index])
                    token_index += 1
            sequences[sequence_index] = replacement^


def _make_byte_tokens() -> List[String]:
    """Build the same GPT-2 byte-to-Unicode map used by ``BpeEncoder``."""
    var tokens = List[String](capacity=BYTE_VOCAB_SIZE)
    var next_codepoint = 0
    for byte_value in range(BYTE_VOCAB_SIZE):
        var visible = (
            (byte_value >= 33 and byte_value < 127) or (byte_value >= 161 and byte_value < 173) or byte_value >= 174
        )
        if visible:
            tokens.append(chr(byte_value))
        else:
            tokens.append(chr(BYTE_VOCAB_SIZE + next_codepoint))
            next_codepoint += 1
    return tokens^
