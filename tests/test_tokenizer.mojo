"""Unit tests for the explicit tokenizer architecture."""

from mojo_tokenizer import (
    BpeTokenizer,
    ModelInputConfig,
    NamedInt64Tensor,
    SpecialTokenSet,
    TokenDecoder,
    TokenizedInputs,
    Vocabulary,
    allocate_int64_buffer,
)
from mojo_tokenizer.preprocessing import LowercaseNormalizer


def make_test_vocabulary() -> Vocabulary:
    var vocabulary = Vocabulary()
    vocabulary.add_token("a", 0)
    vocabulary.add_token("b", 1)
    vocabulary.add_token("ab", 2)
    vocabulary.add_merge("ab", 0)
    return vocabulary^


def make_special_tokens() -> SpecialTokenSet:
    var special_tokens = SpecialTokenSet()
    special_tokens.add("<eos>", 10)
    return special_tokens^


def test_vocabulary() raises:
    var vocabulary = make_test_vocabulary()
    if vocabulary.size() != 3:
        raise Error("Vocabulary must contain three tokens")
    if vocabulary.get_id("ab") != 2:
        raise Error("Merged token must resolve to ID 2")
    if vocabulary.get_merge_rank("ab") != 0:
        raise Error("Merge rank must be preserved")
    if vocabulary.get_id("missing") != -1:
        raise Error("Unknown tokens must resolve to -1")


def test_special_tokens() raises:
    var special_tokens = make_special_tokens()
    var segments = special_tokens.split_on_special("ab<eos>ab")
    if len(segments) != 3:
        raise Error("Special-token scanner must return three segments")
    if segments[1].text != "<eos>" or not segments[1].is_special:
        raise Error("Middle segment must be the registered special token")


def test_encoding() raises:
    var vocabulary = make_test_vocabulary()
    var special_tokens = make_special_tokens()
    var tokenizer = BpeTokenizer(vocabulary, special_tokens)

    var token_ids = tokenizer.encode("ab<eos>")
    if len(token_ids) != 2:
        raise Error("Expected merged token and EOS token")
    if token_ids[0] != 2 or token_ids[1] != 10:
        raise Error("Encoding did not preserve expected token IDs")
    if tokenizer.vocab_size() != 4:
        raise Error("Tokenizer size must include special tokens")

    var empty_ids = tokenizer.encode("")
    if len(empty_ids) != 0:
        raise Error("Empty text must produce no tokens")


def test_decoding() raises:
    var vocabulary = make_test_vocabulary()
    var special_tokens = make_special_tokens()
    var decoder = TokenDecoder(vocabulary, special_tokens)
    var token_ids = List[Int]()
    token_ids.append(2)
    token_ids.append(10)
    if decoder.decode(token_ids) != "ab<eos>":
        raise Error("Diagnostic decoder must reconstruct text")


def test_preprocessing() raises:
    var normalizer = LowercaseNormalizer()
    if normalizer.normalize("AbC") != "abc":
        raise Error("Lowercase normalizer must remain available")


def test_model_input_contracts() raises:
    var config = ModelInputConfig(max_length=2, pad_token_id=0)
    config.set_eos_token(10)

    var shape = List[Int]()
    shape.append(1)
    shape.append(2)
    var values = allocate_int64_buffer(2)
    var value_span = values.unsafe_span()
    value_span[0] = 2
    value_span[1] = 10

    var input_ids = NamedInt64Tensor("input_ids", shape, values^)
    var inputs = TokenizedInputs(input_ids^)
    if inputs.input_ids.name != "input_ids":
        raise Error("Named tensor contract must preserve its input name")
    var input_values = inputs.input_ids.values.unsafe_span()
    if input_values[0] != 2 or input_values[1] != 10:
        raise Error("Allocator-backed tensor values must remain readable")
    if config.max_length != 2:
        raise Error("Model input configuration must preserve max length")


def main() raises:
    test_vocabulary()
    test_special_tokens()
    test_encoding()
    test_decoding()
    test_preprocessing()
    test_model_input_contracts()
    print("All tokenizer tests passed")
