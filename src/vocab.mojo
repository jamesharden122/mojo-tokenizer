"""
Vocabulary management for tokenizers.

This module handles the mapping between tokens (text) and their IDs,
as well as BPE merge rules that define how tokens combine.
"""


struct MergeRule:
    """Represents a BPE merge rule: two tokens merge into one."""

    var first: String
    """First token in the merge pair."""

    var second: String
    """Second token in the merge pair."""

    var result: String
    """Result of merging first + second."""

    var rank: Int
    """Priority rank (lower = higher priority, applied first)."""

    fn __init__(out self, first: String, second: String, rank: Int):
        """Create a new merge rule."""
        self.first = first
        self.second = second
        self.result = first + second
        self.rank = rank


struct Vocabulary:
    """
    Manages the vocabulary mapping between tokens and IDs.

    Stores both the forward mapping (token -> ID) and reverse mapping
    (ID -> token) for efficient lookup in both directions. Also stores
    BPE merge rules for the encoding process.
    """

    var _token_to_id: Dict[String, Int]
    """Map from token text to ID."""

    var _id_to_token: Dict[Int, String]
    """Map from ID to token text."""

    var _merges: Dict[String, Int]
    """Map from merged pair (as string) to rank."""

    var _size: Int
    """Number of tokens in vocabulary."""

    fn __init__(out self):
        """Create an empty vocabulary."""
        self._token_to_id = Dict[String, Int]()
        self._id_to_token = Dict[Int, String]()
        self._merges = Dict[String, Int]()
        self._size = 0

    fn add_token(mut self, token: String, id: Int):
        """
        Add a token to the vocabulary.

        Args:
            token: The token text.
            id: The token ID.
        """
        self._token_to_id[token] = id
        self._id_to_token[id] = token
        self._size += 1

    fn add_merge(mut self, pair: String, rank: Int):
        """
        Add a BPE merge rule.

        Args:
            pair: The concatenated pair (first + second tokens).
            rank: The priority rank (lower = higher priority).
        """
        self._merges[pair] = rank

    fn get_id(self, token: String) -> Int:
        """
        Get the ID for a token.

        Args:
            token: The token text.

        Returns:
            The token ID, or -1 if not found.
        """
        if token in self._token_to_id:
            return self._token_to_id[token]
        return -1

    fn get_text(self, id: Int) -> String:
        """
        Get the text for a token ID.

        Args:
            id: The token ID.

        Returns:
            The token text, or empty string if not found.
        """
        if id in self._id_to_token:
            return self._id_to_token[id]
        return ""

    fn get_merge_rank(self, pair: String) -> Int:
        """
        Get the merge rank for a token pair.

        Args:
            pair: The concatenated pair to look up.

        Returns:
            The merge rank, or -1 if no merge rule exists.
        """
        if pair in self._merges:
            return self._merges[pair]
        return -1

    fn has_token(self, token: String) -> Bool:
        """Check if a token exists in the vocabulary."""
        return token in self._token_to_id

    fn has_id(self, id: Int) -> Bool:
        """Check if an ID exists in the vocabulary."""
        return id in self._id_to_token

    fn size(self) -> Int:
        """Return the number of tokens in the vocabulary."""
        return self._size

    fn clear(mut self):
        """Clear all tokens and merges from the vocabulary."""
        self._token_to_id = Dict[String, Int]()
        self._id_to_token = Dict[Int, String]()
        self._merges = Dict[String, Int]()
        self._size = 0
