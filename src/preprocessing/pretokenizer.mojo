"""
Pre-tokenizers for tokenization pipeline.

Pre-tokenizers split input text into smaller chunks before BPE.
This determines the granularity of the initial tokens:
- Whitespace: splits on whitespace (GPT-2 style)
- ByteLevel: adds special prefix to word starts (GPT-2/GPT-3)
- Punctuation: splits on punctuation characters
- Digits: splits on digit sequences
"""

from .whitespace import is_whitespace, skip_whitespace_simd


struct PreToken(Copyable, Movable):
    """A pre-tokenized chunk with offset information."""

    var text: String
    """The pre-token text."""

    var start: Int
    """Start offset in original text."""

    var end: Int
    """End offset in original text."""

    def __init__(out self, text: String, start: Int, end: Int):
        """Create a pre-token."""
        self.text = text
        self.start = start
        self.end = end

trait PreTokenizer:
    """Split input text into pre-token chunks."""

    def pre_tokenize(self, text: String) -> List[PreToken]:
        ...


struct WhitespacePreTokenizer(PreTokenizer):
    """
    Whitespace pre-tokenizer.

    Splits text on whitespace boundaries. Each word becomes a pre-token.
    Used by BERT and many other models.
    """

    def __init__(out self):
        """Create a whitespace pre-tokenizer."""
        pass

    def pre_tokenize(self, text: String) -> List[PreToken]:
        """Split text on whitespace."""
        var result = List[PreToken]()
        var start = 0
        var n = text.byte_length()

        while start < n:
            # Skip leading whitespace
            start = skip_whitespace_simd(text, start)
            if start >= n:
                break

            # Find end of word
            var end = start
            while end < n:
                var code = ord(text[byte=end])
                if is_whitespace(UInt8(code)):
                    break
                end += 1

            if end > start:
                result.append(PreToken(String(text[byte=start:end]), start, end))

            start = end

        return result^


struct ByteLevelPreTokenizer(PreTokenizer):
    """
    Byte-level pre-tokenizer (GPT-2/GPT-3 style).

    Splits on whitespace and adds a special prefix (Ġ = space) to
    tokens that start after whitespace. This allows the model to
    learn that certain tokens typically start words.

    Example:
        "Hello world" → ["Hello", "Ġworld"]
    """

    var add_prefix_space: Bool
    """Whether to add prefix space to first token."""

    def __init__(out self, add_prefix_space: Bool = True):
        """Create a byte-level pre-tokenizer."""
        self.add_prefix_space = add_prefix_space

    def pre_tokenize(self, text: String) -> List[PreToken]:
        """Split text with byte-level encoding."""
        var result = List[PreToken]()

        if text.byte_length() == 0:
            return result^

        var pos = 0
        var n = text.byte_length()
        var is_first = True

        while pos < n:
            # Find start of next word
            var word_start = pos
            var has_leading_space = False

            # Check for leading whitespace
            if pos < n:
                var code = ord(text[byte=pos])
                if code == 32:  # space
                    has_leading_space = True
                    pos += 1
                    word_start = pos

            # Find end of word (up to next space)
            var word_end = pos
            while word_end < n and ord(text[byte=word_end]) != 32:
                word_end += 1

            if word_end > word_start:
                var word_text = String(text[byte=word_start:word_end])

                # Add prefix Ġ if this word had leading space (or is first with add_prefix_space)
                if has_leading_space or (is_first and self.add_prefix_space):
                    word_text = "Ġ" + word_text

                result.append(PreToken(word_text, word_start, word_end))
                is_first = False

            pos = word_end

        return result^


struct PunctuationPreTokenizer(PreTokenizer):
    """
    Punctuation pre-tokenizer.

    Isolates punctuation characters as separate tokens.
    """

    var behavior: String  # "isolated", "merged_with_previous", "merged_with_next"

    def __init__(out self, behavior: String = "isolated"):
        """Create a punctuation pre-tokenizer."""
        self.behavior = behavior

    def pre_tokenize(self, text: String) -> List[PreToken]:
        """Split text isolating punctuation."""
        var result = List[PreToken]()
        var current_start = 0
        var current_text = String()

        for i in range(text.byte_length()):
            var c = String(text[byte=i])
            var is_punct = self._is_punctuation(c)

            if is_punct:
                # Flush current word if any
                if current_text.byte_length() > 0:
                    result.append(PreToken(current_text, current_start, i))
                    current_text = String()

                # Add punctuation as its own token
                result.append(PreToken(c, i, i + 1))
                current_start = i + 1
            else:
                if current_text.byte_length() == 0:
                    current_start = i
                current_text += c

        # Flush remaining
        if current_text.byte_length() > 0:
            result.append(PreToken(current_text, current_start, text.byte_length()))

        return result^

    def _is_punctuation(self, c: String) -> Bool:
        """Check if character is punctuation."""
        var code = ord(c)
        # ASCII punctuation ranges
        return (code >= 33 and code <= 47) or   # ! " # $ % & ' ( ) * + , - . /
               (code >= 58 and code <= 64) or   # : ; < = > ? @
               (code >= 91 and code <= 96) or   # [ \ ] ^ _ `
               (code >= 123 and code <= 126)    # { | } ~


struct DigitPreTokenizer(PreTokenizer):
    """
    Digit pre-tokenizer.

    Separates digit sequences from other characters.
    """

    var individual_digits: Bool
    """Whether to split into individual digits."""

    def __init__(out self, individual_digits: Bool = False):
        """Create a digit pre-tokenizer."""
        self.individual_digits = individual_digits

    def pre_tokenize(self, text: String) -> List[PreToken]:
        """Split text separating digits."""
        var result = List[PreToken]()
        var current_start = 0
        var current_text = String()
        var in_digits = False

        for i in range(text.byte_length()):
            var c = String(text[byte=i])
            var code = ord(c)
            var is_digit = code >= 48 and code <= 57

            if is_digit != in_digits:
                # Transition - flush current
                if current_text.byte_length() > 0:
                    if in_digits and self.individual_digits:
                        # Split into individual digits
                        for j in range(current_text.byte_length()):
                            result.append(PreToken(
                                String(current_text[byte=j]),
                                current_start + j,
                                current_start + j + 1
                            ))
                    else:
                        result.append(PreToken(current_text, current_start, i))
                    current_text = String()

                current_start = i
                in_digits = is_digit

            current_text += c

        # Flush remaining
        if current_text.byte_length() > 0:
            if in_digits and self.individual_digits:
                for j in range(current_text.byte_length()):
                    result.append(PreToken(
                        String(current_text[byte=j]),
                        current_start + j,
                        current_start + j + 1
                    ))
            else:
                result.append(PreToken(current_text, current_start, text.byte_length()))

        return result^


struct SplitPreTokenizer(PreTokenizer):
    """
    Generic split pre-tokenizer.

    Splits on a specific pattern (character or string).
    """

    var pattern: String
    var behavior: String  # "removed", "isolated", "merged_with_previous", "merged_with_next"

    def __init__(out self, pattern: String, behavior: String = "removed"):
        """Create a split pre-tokenizer."""
        self.pattern = pattern
        self.behavior = behavior

    def pre_tokenize(self, text: String) -> List[PreToken]:
        """Split text on pattern."""
        var result = List[PreToken]()

        if self.pattern.byte_length() == 0 or text.byte_length() == 0:
            result.append(PreToken(text, 0, text.byte_length()))
            return result^

        var current_start = 0
        var i = 0

        while i <= text.byte_length() - self.pattern.byte_length():
            # Check for pattern match
            var is_match = True
            for j in range(self.pattern.byte_length()):
                if String(text[byte=i + j]) != String(self.pattern[byte=j]):
                    is_match = False
                    break

            if is_match:
                # Add text before pattern
                if i > current_start:
                    result.append(PreToken(String(text[byte=current_start:i]), current_start, i))

                # Handle pattern based on behavior
                if self.behavior == "isolated":
                    result.append(PreToken(self.pattern, i, i + self.pattern.byte_length()))
                # "removed" behavior: don't add the pattern

                current_start = i + self.pattern.byte_length()
                i = current_start
            else:
                i += 1

        # Add remaining text
        if current_start < text.byte_length():
            result.append(PreToken(String(text[byte=current_start:]), current_start, text.byte_length()))

        return result^
