"""Special token handling for tokenizers.

Special tokens are tokens with specific meaning that should never be
split during tokenization (e.g., [CLS], [SEP], <|endoftext|>, <s>, </s>).
"""

from std.collections.interval import Interval


struct SpecialToken(Copyable, Movable):
    """Represents a special token."""

    var text: String
    """The token text (e.g., '<|endoftext|>')."""

    var id: Int
    """The token ID."""

    def __init__(out self, text: String, id: Int):
        """Create a new special token."""
        self.text = text
        self.id = id


struct TextSegment(Copyable, Movable):
    """A segment of text, either special or ordinary."""

    var text: String
    """The text content."""

    var is_special: Bool
    """Whether this is a special token."""

    var span: Interval[Int]
    """The segment's half-open UTF-8 byte span in the input."""

    def __init__(out self, text: String, is_special: Bool):
        """Create a new text segment."""
        self.text = text
        self.is_special = is_special
        self.span = Interval(0, text.byte_length())

    def __init__(
        out self,
        text: String,
        is_special: Bool,
        span: Interval[Int],
    ):
        """Create a segment aligned to its original input bytes."""
        self.text = text
        self.is_special = is_special
        self.span = span


struct SpecialTokenSet(Copyable, Movable):
    """
    Manages special tokens for a tokenizer.

    Special tokens are never split during tokenization and are
    looked up directly. Common examples:
    - [CLS], [SEP], [PAD], [MASK] (BERT-style)
    - <|endoftext|>, <|im_start|>, <|im_end|> (GPT-style)
    - <s>, </s>, <unk>, <pad> (SentencePiece-style)
    """

    var _text_to_id: Dict[String, Int]
    """Map from special token text to ID."""

    var _id_to_text: Dict[Int, String]
    """Map from ID to special token text."""

    var _tokens: List[SpecialToken]
    """List of all special tokens (for iteration)."""

    def __init__(out self):
        """Create an empty special tokens manager."""
        self._text_to_id = Dict[String, Int]()
        self._id_to_text = Dict[Int, String]()
        self._tokens = List[SpecialToken]()

    def add(mut self, text: String, id: Int):
        """
        Add a special token.

        Args:
            text: The special token text.
            id: The token ID.
        """
        self._text_to_id[text] = id
        self._id_to_text[id] = text
        self._tokens.append(SpecialToken(text, id))

    def get_id(self, text: String) -> Int:
        """
        Get the ID for a special token.

        Args:
            text: The special token text.

        Returns:
            The token ID, or -1 if not a special token.
        """
        if text in self._text_to_id:
            try:
                return self._text_to_id[text]
            except:
                return -1
        return -1

    def get_text(self, id: Int) -> String:
        """
        Get the text for a special token ID.

        Args:
            id: The token ID.

        Returns:
            The token text, or empty string if not a special token.
        """
        if id in self._id_to_text:
            try:
                return self._id_to_text[id]
            except:
                return ""
        return ""

    def is_special(self, text: String) -> Bool:
        """Check if a token text is a special token."""
        return text in self._text_to_id

    def is_special_id(self, id: Int) -> Bool:
        """Check if a token ID is a special token."""
        return id in self._id_to_text

    def size(self) -> Int:
        """Return the number of special tokens."""
        return len(self._tokens)

    def split_on_special(self, text: String) -> List[TextSegment]:
        """
        Split text on special tokens.

        This is used during encoding to identify special tokens
        that should not be split further.

        Args:
            text: The input text to split.

        Returns:
            List of TextSegments, alternating between ordinary
            text and special tokens.

        Example:
            Input: "Hello <|endoftext|> World"
            Output resembles:
            ```
            [
                TextSegment("Hello ", is_special=False),
                TextSegment("<|endoftext|>", is_special=True),
                TextSegment(" World", is_special=False)
            ]
            ```
        """
        var result = List[TextSegment]()

        if text.byte_length() == 0:
            return result^

        # Simple approach: iterate through text looking for special tokens
        # TODO: Use a more efficient algorithm (e.g., Aho-Corasick)
        var current_pos = 0
        var current_segment_start = 0

        while current_pos < text.byte_length():
            var found_special = False

            # Check if any special token starts at current position
            for i in range(len(self._tokens)):
                var token = self._tokens[i].copy()
                var token_text = token.text
                var token_len = token_text.byte_length()

                if self._matches_at(text, token_text, current_pos):
                    # Found a special token
                    if current_pos > current_segment_start:
                        result.append(
                            TextSegment(
                                String(text[byte=current_segment_start:current_pos]),
                                is_special=False,
                                span=Interval(current_segment_start, current_pos),
                            )
                        )
                    result.append(
                        TextSegment(
                            token_text,
                            is_special=True,
                            span=Interval(current_pos, current_pos + token_len),
                        )
                    )
                    current_pos += token_len
                    current_segment_start = current_pos
                    found_special = True
                    break

            if not found_special:
                current_pos += 1

        # Add any remaining ordinary text
        if current_segment_start < text.byte_length():
            result.append(
                TextSegment(
                    String(text[byte = current_segment_start : text.byte_length()]),
                    is_special=False,
                    span=Interval(current_segment_start, text.byte_length()),
                )
            )

        return result^

    def _matches_at(self, text: String, token: String, byte_offset: Int) -> Bool:
        var token_len = token.byte_length()
        if byte_offset + token_len > text.byte_length():
            return False
        var source = text.unsafe_ptr()
        var target = token.unsafe_ptr()
        for i in range(token_len):
            if source[unsafe_offset=byte_offset + i] != target[unsafe_offset=i]:
                return False
        return True

    def clear(mut self):
        """Clear all special tokens."""
        self._text_to_id = Dict[String, Int]()
        self._id_to_text = Dict[Int, String]()
        self._tokens = List[SpecialToken]()
