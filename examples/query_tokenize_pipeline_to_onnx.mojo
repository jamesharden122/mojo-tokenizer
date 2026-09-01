"""Incomplete fixed-shape query-tokenization pipeline for ONNX export."""

from mojo_tokenizer import BpeTokenizer, OnnxTens, SpecialTokenSet, Vocabulary
from std.memory.alloc import Layout as AllocationLayout, alloc, dealloc


comptime SequenceLength = 8
comptime QueryRows = 4
comptime BatchSize = 2
comptime ElementCount = QueryRows * SequenceLength
comptime PadTokenId = Int64(11)


def query_tokenize_pipeline_to_onnx() raises:
    """Stage one padded query tensor, then hand it to the future ONNX writer."""

    var vocabulary = Vocabulary()
    vocabulary.add_token("a", 0)
    vocabulary.add_token("b", 1)
    vocabulary.add_token("ab", 2)
    vocabulary.add_merge("ab", 0)

    var special_tokens = SpecialTokenSet()
    special_tokens.add("<eos>", 10)
    special_tokens.add("<pad>", Int(PadTokenId))

    var tokenizer = BpeTokenizer(vocabulary, special_tokens)
    var encoded = tokenizer.encode("ab<eos>")
    if len(encoded) > SequenceLength:
        raise Error("query exceeds the fixed ONNX sequence length")

    var storage = alloc(AllocationLayout[Int64](count=ElementCount))
    var storage_ptr = storage.unsafe_ptr()
    for index in range(ElementCount):
        storage_ptr.unsafe_offset(index).unsafe_write(PadTokenId)
    for row in range(QueryRows):
        for index in range(len(encoded)):
            storage.unsafe_span()[row * SequenceLength + index] = Int64(encoded[index])

    var query_token_ids = OnnxTens[QueryRows, BatchSize, SequenceLength, DType.int64](storage.unsafe_span())
    var output = query_token_ids.write_tensor()
    var query_vector = OnnxTens[1, 1, SequenceLength, DType.int64](
        storage.unsafe_span(),
        False,
    )
    var vector_output = query_vector.write_vector()

    var has_mismatch = False
    for index in range(ElementCount):
        if output.unsafe_span()[index] != storage.unsafe_span()[index]:
            has_mismatch = True
    for index in range(SequenceLength):
        if vector_output.unsafe_span()[index] != storage.unsafe_span()[index]:
            has_mismatch = True

    dealloc(vector_output^)
    dealloc(output^)
    dealloc(storage^)
    if has_mismatch:
        raise Error("parallel ONNX tensor copy changed a token value")


def main() raises:
    query_tokenize_pipeline_to_onnx()
