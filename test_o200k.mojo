"""Test loading o200k_base vocabulary."""

from src.vocab import Vocabulary
from src.byte_trie import ByteTrie
from src.fast_encoder import fast_encode
from src.flat_token_storage import FlatTokenStorage, flat_decode
from src.pretokenizer import encode_with_pretokenizer
from memory import UnsafePointer
from time import perf_counter_ns


fn _base64_char_value(c: UInt8) -> Int:
    if c >= ord("A") and c <= ord("Z"):
        return Int(c) - ord("A")
    elif c >= ord("a") and c <= ord("z"):
        return Int(c) - ord("a") + 26
    elif c >= ord("0") and c <= ord("9"):
        return Int(c) - ord("0") + 52
    elif c == ord("+"):
        return 62
    elif c == ord("/"):
        return 63
    return 0


fn base64_decode(encoded: String) -> List[UInt8]:
    var result = List[UInt8]()
    var enc_bytes = encoded.as_bytes()
    var length = len(enc_bytes)
    var pad = 0
    if length > 0 and enc_bytes[length - 1] == ord("="):
        pad += 1
    if length > 1 and enc_bytes[length - 2] == ord("="):
        pad += 1
    var i = 0
    while i < length - pad:
        var val0 = _base64_char_value(enc_bytes[i]) if i < length else 0
        var val1 = _base64_char_value(enc_bytes[i + 1]) if i + 1 < length else 0
        var val2 = _base64_char_value(enc_bytes[i + 2]) if i + 2 < length else 0
        var val3 = _base64_char_value(enc_bytes[i + 3]) if i + 3 < length else 0
        var combined = (val0 << 18) | (val1 << 12) | (val2 << 6) | val3
        result.append(UInt8((combined >> 16) & 0xFF))
        if i + 2 < length - pad:
            result.append(UInt8((combined >> 8) & 0xFF))
        if i + 3 < length - pad:
            result.append(UInt8(combined & 0xFF))
        i += 4
    return result^


fn bytes_to_bpe_string(data: List[UInt8]) -> String:
    var result = String()
    var n = 0
    var byte_to_unicode = List[Int](capacity=256)
    for i in range(256):
        if i >= 33 and i <= 126:
            byte_to_unicode.append(i)
        else:
            byte_to_unicode.append(256 + n)
            n += 1
    for i in range(len(data)):
        var byte_val = Int(data[i])
        result += chr(byte_to_unicode[byte_val])
    return result


fn read_file_string(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


fn load_vocab(path: String) raises -> Tuple[Vocabulary, ByteTrie]:
    var vocab = Vocabulary()
    var trie = ByteTrie()
    var content = read_file_string(path)
    var lines = content.split("\n")
    for i in range(len(lines)):
        var line = lines[i]
        if len(line) == 0:
            continue
        var last_space = -1
        for j in range(len(line) - 1, -1, -1):
            if line[j] == " ":
                last_space = j
                break
        if last_space < 0:
            continue
        var encoded_token = String(line[:last_space])
        var rank_str = String(line[last_space + 1 :])
        if len(encoded_token) == 0 or len(rank_str) == 0:
            continue
        var rank = atol(rank_str)
        var token_bytes = base64_decode(encoded_token)
        var token = bytes_to_bpe_string(token_bytes)
        vocab.add_token_bytes(token, rank, token_bytes)
        trie.insert(token_bytes, rank)
    vocab.build_backtrack_tables()
    return Tuple(vocab^, trie^)


fn main() raises:
    print("Loading o200k_base vocabulary...")
    var start = perf_counter_ns()
    var vocab_trie = load_vocab("data/o200k_base.tiktoken")
    var vocab = vocab_trie[0].copy()
    var trie = vocab_trie[1].copy()
    var load_time = (perf_counter_ns() - start) // 1_000_000
    print("  Load time:", load_time, "ms")
    
    # Build flat storage for decoding
    var flat_storage = FlatTokenStorage.from_nested_lists(vocab._id_to_bytes)
    print("  Vocab size:", flat_storage.token_count)
    
    # Test encoding
    var test_text = "Hello, world!"
    var text_bytes = test_text.as_bytes()
    var text_ptr = text_bytes.unsafe_ptr()
    var text_len = len(text_bytes)
    
    print("\nTest: \"Hello, world!\"")
    var tokens = encode_with_pretokenizer(vocab, trie, text_ptr, text_len)
    print("  Tokens:", end="")
    for i in range(len(tokens)):
        print(" ", tokens[i], end="")
    print()
    print("  Expected (o200k_base): 13225 11 2375 0")
    
    # Test decoding roundtrip
    var decoded = flat_decode(flat_storage, tokens)
    print("  Decoded length:", len(decoded), "bytes")
    print("  Original length:", text_len, "bytes")
    print("  Roundtrip:", "PASS" if len(decoded) == text_len else "FAIL")
