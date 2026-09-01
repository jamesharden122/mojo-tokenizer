"""Unit tests for the explicit tokenizer architecture."""

from mojo_tokenizer import (
    BpeTokenizer,
    BpeTrainer,
    BytePatternSet,
    SpecialTokenSet,
    TokenDecoder,
    Vocabulary,
)
from mojo_tokenizer.core import BitField, CpuBacktrackBatch
from mojo_tokenizer.core.byte_trie import ByteTrie
from std.collections.interval import Interval


def make_test_vocabulary() -> Vocabulary:
    var vocabulary = Vocabulary()
    vocabulary.add_token("a", 0)
    vocabulary.add_token("b", 1)
    vocabulary.add_token("ab", 2)
    vocabulary.add_merge("ab", 0)
    return vocabulary^


def make_test_trie() -> ByteTrie:
    var trie = ByteTrie()
    trie.insert_string("a", 0)
    trie.insert_string("b", 1)
    trie.insert_string("ab", 2)
    trie.freeze()
    return trie^


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


def test_raw_vocabulary_byte_length() raises:
    var vocabulary = Vocabulary()
    var raw_bytes = List[UInt8]()
    raw_bytes.append(UInt8(0))
    raw_bytes.append(UInt8(255))
    vocabulary.add_token_bytes("raw", 0, raw_bytes)
    if vocabulary.get_byte_length(0) != 2:
        raise Error("Backtracking must use a raw vocabulary token's byte length")


def test_special_tokens() raises:
    var special_tokens = make_special_tokens()
    var segments = special_tokens.split_on_special("ab<eos>ab")
    if len(segments) != 3:
        raise Error("Special-token scanner must return three segments")
    if segments[1].text != "<eos>" or not segments[1].is_special:
        raise Error("Middle segment must be the registered special token")
    if segments[0].span.start != 0 or segments[0].span.end != 2:
        raise Error("Ordinary special-token segments must retain byte offsets")
    if segments[1].span.start != 2 or segments[1].span.end != 7:
        raise Error("Special tokens must retain their full byte interval")


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

    var token_spans = tokenizer.encode_with_spans("ab<eos>ab")
    if len(token_spans) != 3:
        raise Error("Span encoding must preserve ordinary and special tokens")
    if token_spans[0].token_id != 2 or token_spans[0].span.start != 0 or token_spans[0].span.end != 2:
        raise Error("Merged tokens must retain their source byte interval")
    if token_spans[1].token_id != 10 or token_spans[1].span.start != 2 or token_spans[1].span.end != 7:
        raise Error("Special-token spans must use original input offsets")
    if token_spans[2].token_id != 2 or token_spans[2].span.start != 7 or token_spans[2].span.end != 9:
        raise Error("Encoding after a special token must restore the global byte offset")

    var cached_spans = tokenizer.encode_with_spans("ab<eos>ab")
    if cached_spans[2].span.start != 7 or cached_spans[2].span.end != 9:
        raise Error("Cached relative spans must be shifted back to input offsets")


def test_backtracking_encoding() raises:
    var vocabulary = make_test_vocabulary()
    var special_tokens = make_special_tokens()
    var tokenizer = BpeTokenizer(vocabulary, special_tokens)
    tokenizer.set_backtracking_enabled(True)
    tokenizer.reserve_backtracking_capacity(512)
    if tokenizer.backtracking_scratch_capacity() != 512:
        raise Error("Tokenizer must retain its pre-sized CPU backtracking slot")

    var token_ids = tokenizer.encode("abab")
    if len(token_ids) != 2 or token_ids[0] != 2 or token_ids[1] != 2:
        raise Error("Backtracking encoder must preserve allocation-backed token IDs")

    var reused_ids = tokenizer.encode("a")
    if len(reused_ids) != 1 or reused_ids[0] != 0:
        raise Error("Backtracking scratch must reset before the next input")

    var token_spans = tokenizer.encode_with_spans("abab")
    if len(token_spans) != 2:
        raise Error("Backtracking span encoding must preserve token count")
    if token_spans[0].span.start != 0 or token_spans[0].span.end != 2:
        raise Error("Backtracking must retain the first accepted token span")
    if token_spans[1].span.start != 2 or token_spans[1].span.end != 4:
        raise Error("Backtracking must retain the second accepted token span")


def test_bpe_trainer() raises:
    var chunks = List[String]()
    chunks.append("abab")
    chunks.append("abab")
    var trainer = BpeTrainer(target_vocab_size=257, min_pair_frequency=1)
    var vocabulary = trainer.train(chunks)

    if vocabulary.size() != 257 or vocabulary.get_id("ab") != 256:
        raise Error("Trainer must add the most frequent byte pair deterministically")
    var raw_bytes = vocabulary.get_bytes(256)
    if len(raw_bytes) != 2 or raw_bytes[0] != 97 or raw_bytes[1] != 98:
        raise Error("Trainer must retain exact raw bytes for learned tokens")

    var tokenizer = BpeTokenizer(vocabulary, SpecialTokenSet())
    var token_ids = tokenizer.encode("abab")
    if len(token_ids) != 2 or token_ids[0] != 256 or token_ids[1] != 256:
        raise Error("Trainer output must be consumable by the BPE encoder")


def test_utf8_byte_spans() raises:
    var trainer = BpeTrainer(target_vocab_size=256)
    var vocabulary = trainer.train(List[String]())
    var tokenizer = BpeTokenizer(vocabulary, SpecialTokenSet())
    var spans = tokenizer.encode_with_spans("é")
    if len(spans) != 2:
        raise Error("A two-byte UTF-8 code point must produce two base-byte tokens")
    if spans[0].span.start != 0 or spans[0].span.end != 1:
        raise Error("UTF-8 token spans must use byte offsets, not code-point offsets")
    if spans[1].span.start != 1 or spans[1].span.end != 2:
        raise Error("UTF-8 continuation bytes must retain their own byte span")


def test_decoding() raises:
    var vocabulary = make_test_vocabulary()
    var special_tokens = make_special_tokens()
    var decoder = TokenDecoder(vocabulary, special_tokens)
    var token_ids = List[Int]()
    token_ids.append(2)
    token_ids.append(10)
    if decoder.decode(token_ids) != "ab<eos>":
        raise Error("Diagnostic decoder must reconstruct text")


def test_bitfield() raises:
    var field = BitField(130)
    field.clear(0)
    field.clear(1)
    field.clear(129)
    if field.is_set(0) or field.is_set(1) or field.is_set(129):
        raise Error("BitField clear must unset the requested bit")
    if field.successor(0) != 2:
        raise Error("BitField successor must find the next set bit")
    if field.predecessor(129) != 128:
        raise Error("BitField predecessor must find the prior set bit")
    field.reset_all()
    if not field.is_set(0) or not field.is_set(129):
        raise Error("BitField reset_all must restore all reachability bits")


def test_cpu_backtrack_batch_reuses_slots() raises:
    var vocabulary = make_test_vocabulary()
    var trie = make_test_trie()
    var batch = CpuBacktrackBatch[2](2)

    var first = batch.encode_slot(0, vocabulary, trie, "abab")
    if len(first) != 2 or first[0] != 2 or first[1] != 2:
        raise Error("Batch slot must encode its first input")
    if batch.slot_capacity() < 4:
        raise Error("Batch slots must grow together for larger inputs")

    var reused = batch.encode_slot(0, vocabulary, trie, "a")
    if len(reused) != 1 or reused[0] != 0:
        raise Error("Reused batch slot must reset token and reachability state")

    var second_slot = batch.encode_slot(1, vocabulary, trie, "b")
    if len(second_slot) != 1 or second_slot[0] != 1:
        raise Error("Each batch slot must retain independent reusable storage")

    var text_bytes = List[UInt8]()
    text_bytes.append(UInt8(97))
    text_bytes.append(UInt8(98))
    var byte_input = batch.encode_slot_bytes(1, vocabulary, trie, text_bytes)
    if len(byte_input) != 1 or byte_input[0] != 2:
        raise Error("Batch slots must accept byte inputs without a String conversion")


def test_byte_trie_interval_graph() raises:
    var trie = ByteTrie()
    for i in range(20):
        var key = List[UInt8]()
        key.append(UInt8(i))
        trie.insert(key, i)

    var overflow_child_key = List[UInt8]()
    overflow_child_key.append(UInt8(19))

    var overflow_node_key = List[UInt8]()
    for _ in range(70):
        overflow_node_key.append(UInt8(200))
    trie.insert(overflow_node_key, 70)
    trie.freeze()
    if trie.lookup_exact(overflow_child_key) != 19:
        raise Error("ByteTrie must find high-fanout graph edges")
    if trie.lookup_exact(overflow_node_key) != 70 or trie.node_count() < 71:
        raise Error("ByteTrie must find long graph paths after freezing")
    if trie.nodes[0].tree_index < 0 or len(trie.interval_trees) != 1:
        raise Error("High-fanout prefixes must promote to a per-prefix IntervalTree")
    if trie.edges[0].bytes.start != 0 or trie.edges[0].bytes.end != 1:
        raise Error("Exact trie bytes must freeze as singleton intervals")


def test_byte_trie_ranged_edges() raises:
    var trie = ByteTrie()
    var digits = List[Interval[Int]]()
    digits.append(Interval(48, 58))
    trie.insert_intervals(digits, 7)
    trie.freeze()

    var digit = List[UInt8]()
    digit.append(UInt8(53))
    if trie.lookup_exact(digit) != 7:
        raise Error("A ranged edge must match a contained byte")

    var letter = List[UInt8]()
    letter.append(UInt8(97))
    if trie.lookup_exact(letter) >= 0:
        raise Error("A ranged edge must reject bytes outside its interval")


def test_byte_trie_per_prefix_trees() raises:
    var trie = ByteTrie()
    for prefix in range(2):
        for suffix in range(8):
            var token = List[UInt8]()
            token.append(UInt8(97 + prefix))
            token.append(UInt8(suffix))
            trie.insert(token, prefix * 8 + suffix)
    trie.freeze()
    if len(trie.interval_trees) != 2:
        raise Error("Each sufficiently wide prefix must own a separate IntervalTree")

    var first = List[UInt8]()
    first.append(UInt8(97))
    first.append(UInt8(7))
    var second = List[UInt8]()
    second.append(UInt8(98))
    second.append(UInt8(7))
    if trie.lookup_exact(first) != 7 or trie.lookup_exact(second) != 15:
        raise Error("Per-prefix trees must isolate identical child-byte keys")


def test_byte_patterns() raises:
    var vocabulary = make_test_vocabulary()
    vocabulary.add_token("<digit>", 3)
    vocabulary.add_token("<two-digits>", 4)
    var intervals = List[Interval[Int]]()
    intervals.append(Interval(48, 58))
    var patterns = BytePatternSet()
    patterns.add(intervals, 3)
    var two_digits = intervals.copy()
    two_digits.append(Interval(48, 58))
    patterns.add(two_digits, 4)

    var tokenizer = BpeTokenizer(vocabulary, make_special_tokens(), patterns)
    var spans = tokenizer.encode_with_spans("ab<eos>5ab")
    if len(spans) != 4:
        raise Error("Pattern matching must preserve surrounding BPE and special tokens")
    if spans[0].token_id != 2 or spans[0].span.start != 0 or spans[0].span.end != 2:
        raise Error("Ordinary text before a pattern must retain its span")
    if spans[1].token_id != 10 or spans[1].span.start != 2 or spans[1].span.end != 7:
        raise Error("Special tokens must take precedence over byte patterns")
    if spans[2].token_id != 3 or spans[2].span.start != 7 or spans[2].span.end != 8:
        raise Error("A ranged pattern must emit its canonical vocabulary ID and source span")
    if spans[3].token_id != 2 or spans[3].span.start != 8 or spans[3].span.end != 10:
        raise Error("Ordinary BPE must resume after a ranged pattern")

    var ids = tokenizer.encode("ab<eos>5ab")
    if len(ids) != len(spans):
        raise Error("ID encoding must project exactly from span encoding")
    for i in range(len(ids)):
        if ids[i] != spans[i].token_id:
            raise Error("ID and span encoding results cannot diverge")

    var longest = tokenizer.encode_with_spans("55")
    if len(longest) != 1 or longest[0].token_id != 4:
        raise Error("Pattern lookup must prefer the longest terminal range path")
    if longest[0].span.start != 0 or longest[0].span.end != 2:
        raise Error("A multi-byte ranged pattern must cover its complete byte span")

    var overlapping = List[Interval[Int]]()
    overlapping.append(Interval(53, 64))
    var rejected = False
    try:
        patterns.add(overlapping, 3)
    except:
        rejected = True
    if not rejected:
        raise Error("Pattern registration must reject overlapping sibling intervals")


def test_byte_trie_unused_inline_lanes_are_not_matches() raises:
    var trie = ByteTrie()
    var key = List[UInt8]()
    key.append(UInt8(97))
    trie.insert(key, 97)
    trie.freeze()

    var zero_byte = List[UInt8]()
    zero_byte.append(UInt8(0))
    if trie.lookup_exact(zero_byte) >= 0:
        raise Error("ByteTrie must ignore unused inline lanes for byte zero")


def main() raises:
    test_vocabulary()
    test_raw_vocabulary_byte_length()
    test_special_tokens()
    test_encoding()
    test_backtracking_encoding()
    test_bpe_trainer()
    test_utf8_byte_spans()
    test_decoding()
    test_bitfield()
    test_cpu_backtrack_batch_reuses_slots()
    test_byte_trie_interval_graph()
    test_byte_trie_ranged_edges()
    test_byte_trie_per_prefix_trees()
    test_byte_trie_unused_inline_lanes_are_not_matches()
    test_byte_patterns()
    print("All tokenizer tests passed")
