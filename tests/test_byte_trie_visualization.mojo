from mojo_tokenizer.core.byte_trie import ByteTrie


def print_edge(trie: ByteTrie, depth: Int, edge_index: Int):
    var edge = trie.edges[edge_index].copy()
    var indent = String()
    for _ in range(depth):
        indent += "  "

    var label = "[" + String(edge.bytes.start) + ", " + String(edge.bytes.end) + ")"
    if len(edge.bytes) == 1:
        label = "'" + String(chr(edge.bytes.start)) + "'"

    if trie.nodes[edge.child].terminal:
        print(indent, "└─ ", label, " -> node ", edge.child, " [token ", trie._token_id_for_terminal_edge(edge_index), "]")
    else:
        print(indent, "└─ ", label, " -> node ", edge.child)

    print_node(trie, edge.child, depth + 1)


def print_node(trie: ByteTrie, node_index: Int, depth: Int):
    var interval = trie.nodes[node_index].edges
    for edge_index in range(interval.start, interval.end):
        print_edge(trie, depth, edge_index)


def main() raises:
    var trie = ByteTrie()
    trie.insert_string("a", 0)
    trie.insert_string("an", 1)
    trie.insert_string("and", 2)
    trie.insert_string("ant", 3)
    trie.insert_string("any", 4)
    trie.insert_string("app", 5)
    trie.insert_string("apple", 6)
    trie.insert_string("apply", 7)
    trie.insert_string("ape", 8)
    trie.insert_string("bat", 9)
    trie.insert_string("bath", 10)
    trie.insert_string("batch", 11)
    trie.insert_string("cat", 12)
    trie.insert_string("car", 13)
    trie.insert_string("cart", 14)
    trie.insert_string("dog", 15)
    trie.insert_string("do", 16)
    trie.insert_string("done", 17)
    trie.insert_string("<eos>", 18)
    trie.insert_string("<pad>", 19)
    trie.freeze()

    print("ByteTrie: 20 vocabulary tokens, ", trie.node_count(), " nodes")
    print("root")
    print_node(trie, 0, 0)
