"""Lazy runtime byte trie stored as an interval-indexed graph."""

from std.collections.interval import Interval, IntervalTree


comptime INLINE_TOKEN_BYTE_CAPACITY: Int = 256
comptime INTERVAL_TREE_MIN_EDGES: Int = 8


struct ByteTrieNode(Copyable, Movable):
    """A graph node whose outgoing edges occupy ``edges`` in ``ByteTrie``."""

    var edges: Interval[Int]
    var tree_index: Int
    var terminal: Bool

    def __init__(
        out self,
        edges: Interval[Int],
        tree_index: Int = -1,
        terminal: Bool = False,
    ):
        self.edges = edges
        self.tree_index = tree_index
        self.terminal = terminal


struct ByteTrieEdge(Copyable, Movable):
    """An interval-labeled directed edge from the current node to ``child``."""

    var bytes: Interval[Int]
    var child: Int

    def __init__(out self, bytes: Interval[Int], child: Int):
        self.bytes = bytes
        self.child = child


struct _ByteTrieBuildNode(Copyable, Movable):
    """Mutable construction state that is flattened when the trie freezes."""

    var edges: List[ByteTrieEdge]
    var terminal: Bool
    var token_id: Int

    def __init__(out self):
        self.edges = List[ByteTrieEdge]()
        self.terminal = False
        self.token_id = -1


struct ByteTrie(Movable):
    """
    A vocabulary trie with a mutable builder and a frozen interval graph.

    ``insert*`` methods update the builder. Call ``freeze()`` once after
    vocabulary construction. Runtime lookup then traverses contiguous edge
    intervals, while ``tokens_by_terminal_edge`` maps a terminal edge to its
    token ID.
    """

    var nodes: List[ByteTrieNode]
    var edges: List[ByteTrieEdge]
    var interval_trees: List[IntervalTree[Int, Int]]
    var tokens_by_terminal_edge: Dict[Int, Int]
    var _build_nodes: List[_ByteTrieBuildNode]
    var _root_token_id: Int
    var _size: Int
    var _frozen: Bool

    def __init__(out self):
        self.nodes = List[ByteTrieNode]()
        self.edges = List[ByteTrieEdge]()
        self.interval_trees = List[IntervalTree[Int, Int]]()
        self.tokens_by_terminal_edge = Dict[Int, Int]()
        self._build_nodes = List[_ByteTrieBuildNode]()
        self._build_nodes.append(_ByteTrieBuildNode())
        self._root_token_id = -1
        self._size = 0
        self._frozen = False

    def insert(mut self, token_bytes: List[UInt8], token_id: Int):
        """Add a token to the mutable graph builder."""
        assert not self._frozen, "ByteTrie cannot be modified after freeze"
        var node_index = 0
        for byte_index in range(len(token_bytes)):
            var byte = Int(token_bytes[byte_index])
            node_index = self._find_or_add_child(node_index, Interval(byte, byte + 1))
        self._set_terminal(node_index, token_id)

    def insert_inline[
        Capacity: Int
    ](mut self, token_bytes: InlineArray[UInt8, Capacity], token_length: Int, token_id: Int,):
        """Add a bounded token without allocating a temporary byte list."""
        assert not self._frozen, "ByteTrie cannot be modified after freeze"
        assert token_length >= 0 and token_length <= Capacity, "Inline token length out of bounds"
        var node_index = 0
        for byte_index in range(token_length):
            var byte = Int(token_bytes[byte_index])
            node_index = self._find_or_add_child(node_index, Interval(byte, byte + 1))
        self._set_terminal(node_index, token_id)

    def insert_intervals(
        mut self,
        byte_intervals: List[Interval[Int]],
        token_id: Int,
    ):
        """Add a deterministic byte-range path to the mutable graph builder."""
        assert not self._frozen, "ByteTrie cannot be modified after freeze"
        assert len(byte_intervals) > 0, "ByteTrie interval path cannot be empty"
        var node_index = 0
        for interval_index in range(len(byte_intervals)):
            var interval = byte_intervals[interval_index]
            self._validate_byte_interval(interval)
            node_index = self._find_or_add_child(node_index, interval)
        self._set_terminal(node_index, token_id)

    def insert_string(mut self, token: String, token_id: Int):
        """Add a bounded UTF-8 token through a fixed stack-resident buffer."""
        var token_length = token.byte_length()
        assert token_length <= INLINE_TOKEN_BYTE_CAPACITY, "Token exceeds inline trie insertion capacity"

        var bytes = InlineArray[UInt8, INLINE_TOKEN_BYTE_CAPACITY](fill=0)
        var source = token.unsafe_ptr()
        for byte_index in range(token_length):
            bytes[byte_index] = source[unsafe_offset=byte_index]
        self.insert_inline(bytes, token_length, token_id)

    def freeze(mut self):
        """Flatten builder adjacency lists into the immutable runtime graph."""
        if self._frozen:
            return

        self.nodes = List[ByteTrieNode](capacity=len(self._build_nodes))
        self.edges = List[ByteTrieEdge]()
        self.interval_trees = List[IntervalTree[Int, Int]]()
        self.tokens_by_terminal_edge = Dict[Int, Int]()
        var incoming_edges = List[Int](length=len(self._build_nodes), fill=-1)

        for node_index in range(len(self._build_nodes)):
            var build_node = self._build_nodes[node_index].copy()
            self._sort_edges(build_node.edges)
            var start = len(self.edges)
            for edge_index in range(len(build_node.edges)):
                var edge = build_node.edges[edge_index].copy()
                incoming_edges[edge.child] = len(self.edges)
                self.edges.append(edge^)

            var tree_index = -1
            if len(build_node.edges) >= INTERVAL_TREE_MIN_EDGES:
                tree_index = len(self.interval_trees)
                var tree = IntervalTree[Int, Int]()
                for edge_index in range(start, len(self.edges)):
                    tree.insert(self.edges[edge_index].bytes, edge_index)
                self.interval_trees.append(tree^)
            self.nodes.append(ByteTrieNode(Interval(start, len(self.edges)), tree_index, build_node.terminal))

        for node_index in range(1, len(self._build_nodes)):
            var build_node = self._build_nodes[node_index].copy()
            if build_node.terminal:
                self.tokens_by_terminal_edge[incoming_edges[node_index]] = build_node.token_id

        self._build_nodes.clear()
        self._frozen = True

    def is_frozen(self) -> Bool:
        return self._frozen

    @always_inline
    def _child_edge_index(self, node_index: Int, byte: UInt8) -> Int:
        self._require_frozen()
        var node = self.nodes[node_index].copy()
        var byte_value = Int(byte)
        if node.tree_index >= 0:
            try:
                var matches = self.interval_trees[node.tree_index].search(Interval(byte_value, byte_value + 1))
                if len(matches) > 0:
                    return matches[0]
            except:
                pass
            return -1

        for edge_index in range(node.edges.start, node.edges.end):
            if byte_value in self.edges[edge_index].bytes:
                return edge_index
        return -1

    @always_inline
    def _child_index(self, node_index: Int, byte: UInt8) -> Int:
        var edge_index = self._child_edge_index(node_index, byte)
        if edge_index < 0:
            return -1
        return self.edges[edge_index].child

    @always_inline
    def _is_terminal(self, node_index: Int) -> Bool:
        self._require_frozen()
        return self.nodes[node_index].terminal

    def _token_id_for_terminal_edge(self, edge_index: Int) -> Int:
        self._require_frozen()
        if edge_index in self.tokens_by_terminal_edge:
            try:
                return self.tokens_by_terminal_edge[edge_index]
            except:
                return -1
        return -1

    def lookup(self, input_bytes: List[UInt8]) -> TrieLookupResult:
        return self.lookup_at_offset(input_bytes, 0)

    def lookup_at_offset(self, input_bytes: List[UInt8], offset: Int) -> TrieLookupResult:
        self._require_frozen()
        var node_index = 0
        var last_match_id = -1
        var last_match_length = 0

        for byte_index in range(offset, len(input_bytes)):
            var edge_index = self._child_edge_index(node_index, input_bytes[byte_index])
            if edge_index < 0:
                break
            node_index = self.edges[edge_index].child
            if self._is_terminal(node_index):
                last_match_id = self._token_id_for_terminal_edge(edge_index)
                last_match_length = byte_index - offset + 1

        return TrieLookupResult(last_match_id >= 0, last_match_id, last_match_length)

    def lookup_string_at_offset(self, text: String, offset: Int) -> TrieLookupResult:
        """Find the longest match in a string without materializing a byte list."""
        self._require_frozen()
        var node_index = 0
        var last_match_id = -1
        var last_match_length = 0
        var source = text.unsafe_ptr()

        for byte_index in range(offset, text.byte_length()):
            var edge_index = self._child_edge_index(node_index, source[unsafe_offset=byte_index])
            if edge_index < 0:
                break
            node_index = self.edges[edge_index].child
            if self._is_terminal(node_index):
                last_match_id = self._token_id_for_terminal_edge(edge_index)
                last_match_length = byte_index - offset + 1

        return TrieLookupResult(last_match_id >= 0, last_match_id, last_match_length)

    def lookup_exact(self, input_bytes: List[UInt8]) -> Int:
        self._require_frozen()
        var node_index = 0
        var last_edge_index = -1
        for byte_index in range(len(input_bytes)):
            last_edge_index = self._child_edge_index(node_index, input_bytes[byte_index])
            if last_edge_index < 0:
                return -1
            node_index = self.edges[last_edge_index].child

        if node_index == 0:
            return self._root_token_id
        if self._is_terminal(node_index):
            return self._token_id_for_terminal_edge(last_edge_index)
        return -1

    def greedy_tokenize(self, input_bytes: List[UInt8]) -> List[Int]:
        self._require_frozen()
        var result = List[Int]()
        var position = 0
        while position < len(input_bytes):
            var lookup_result = self.lookup_at_offset(input_bytes, position)
            if lookup_result.found:
                result.append(lookup_result.token_id)
                position += lookup_result.match_length
            else:
                result.append(-1)
                position += 1
        return result^

    def size(self) -> Int:
        return self._size

    def node_count(self) -> Int:
        if self._frozen:
            return len(self.nodes)
        return len(self._build_nodes)

    def _find_or_add_child(mut self, node_index: Int, bytes: Interval[Int]) -> Int:
        var build_node = self._build_nodes[node_index].copy()
        for edge_index in range(len(build_node.edges)):
            var existing = build_node.edges[edge_index].bytes
            if existing.start == bytes.start and existing.end == bytes.end:
                return build_node.edges[edge_index].child
            assert not existing.overlaps(bytes), "ByteTrie sibling byte intervals cannot overlap"

        var child = len(self._build_nodes)
        self._build_nodes.append(_ByteTrieBuildNode())
        build_node.edges.append(ByteTrieEdge(bytes, child))
        self._build_nodes[node_index] = build_node^
        return child

    def _validate_byte_interval(self, interval: Interval[Int]):
        assert interval.start >= 0, "ByteTrie interval start must be nonnegative"
        assert interval.end <= 256, "ByteTrie interval end must not exceed 256"
        assert interval.start < interval.end, "ByteTrie interval cannot be empty"

    def _sort_edges(self, mut edges: List[ByteTrieEdge]):
        """Sort one node's small builder edge list by interval start."""
        for i in range(1, len(edges)):
            var edge = edges[i].copy()
            var j = i
            while j > 0 and edges[j - 1].bytes.start > edge.bytes.start:
                edges[j] = edges[j - 1].copy()
                j -= 1
            edges[j] = edge^

    def _set_terminal(mut self, node_index: Int, token_id: Int):
        if node_index == 0:
            self._root_token_id = token_id
        else:
            var build_node = self._build_nodes[node_index].copy()
            build_node.terminal = True
            build_node.token_id = token_id
            self._build_nodes[node_index] = build_node^
        self._size += 1

    def _require_frozen(self):
        assert self._frozen, "Call ByteTrie.freeze() before runtime lookup"


struct TrieLookupResult(Copyable, Movable):
    var found: Bool
    var token_id: Int
    var match_length: Int

    def __init__(out self, found: Bool, token_id: Int, match_length: Int):
        self.found = found
        self.token_id = token_id
        self.match_length = match_length
