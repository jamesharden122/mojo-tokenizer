"""Compare flat and per-prefix IntervalTree byte-interval dispatch."""

from mojo_tokenizer.core.byte_trie import ByteTrieEdge
from std.collections.interval import Interval, IntervalTree
from std.time import perf_counter


comptime ITERATIONS: Int = 200_000


def make_edges(edge_count: Int) -> List[ByteTrieEdge]:
    var edges = List[ByteTrieEdge](capacity=edge_count)
    for byte in range(edge_count):
        edges.append(ByteTrieEdge(Interval(byte, byte + 1), byte))
    return edges^


def make_tree(edges: List[ByteTrieEdge]) -> IntervalTree[Int, Int]:
    var tree = IntervalTree[Int, Int]()
    for edge_index in range(len(edges)):
        tree.insert(edges[edge_index].bytes, edge_index)
    return tree^


def measure_linear(edges: List[ByteTrieEdge], byte: Int) -> Tuple[Float64, Int]:
    var checksum = 0
    var start = perf_counter()
    for _ in range(ITERATIONS):
        for edge_index in range(len(edges)):
            if byte in edges[edge_index].bytes:
                checksum += edge_index
                break
    var elapsed = perf_counter() - start
    return (elapsed * 1_000_000_000.0 / Float64(ITERATIONS), checksum)


def measure_tree(tree: IntervalTree[Int, Int], byte: Int) raises -> Tuple[Float64, Int]:
    var checksum = 0
    var start = perf_counter()
    for _ in range(ITERATIONS):
        var matches = tree.search(Interval(byte, byte + 1))
        checksum += matches[0]
    var elapsed = perf_counter() - start
    return (elapsed * 1_000_000_000.0 / Float64(ITERATIONS), checksum)


def report(edge_count: Int) raises:
    var edges = make_edges(edge_count)
    var tree = make_tree(edges)
    var linear = measure_linear(edges, edge_count - 1)
    var indexed = measure_tree(tree, edge_count - 1)
    print(edge_count, "edges | flat ns:", linear[0], "| IntervalTree ns:", indexed[0])
    if linear[1] != indexed[1]:
        raise Error("Dispatch benchmark checksum mismatch")


def main() raises:
    report(8)
    report(16)
    report(32)
    report(64)
    report(128)
    report(256)
