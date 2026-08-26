"""Programmatic vocabulary and clean tokenizer API example."""

from mojo_tokenizer import BpeTokenizer, SpecialTokenSet, Vocabulary


def main() raises:
    var vocabulary = Vocabulary()
    vocabulary.add_token("a", 0)
    vocabulary.add_token("b", 1)
    vocabulary.add_token("ab", 2)
    vocabulary.add_merge("ab", 0)

    var special_tokens = SpecialTokenSet()
    special_tokens.add("<eos>", 10)

    var tokenizer = BpeTokenizer(vocabulary, special_tokens)
    var token_ids = tokenizer.encode("ab<eos>")
    print("Encoded token IDs:")
    for i in range(len(token_ids)):
        print(token_ids[i])
