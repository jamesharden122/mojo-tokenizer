"""
Explicit text-normalization stages.

Normalizers transform input text before tokenization. Common normalizations:
- NFC/NFKC/NFKD unicode normalization
- Lowercase conversion
- Whitespace normalization
- Accent stripping
"""


trait Normalizer:
    """Base trait for text normalizers."""

    def normalize(self, text: String) -> String:
        """Normalize the input text."""
        ...


struct NFCNormalizer(Normalizer):
    """
    NFC (Canonical Decomposition, followed by Canonical Composition) normalizer.

    Combines characters into their composed form (e.g., é instead of e + ́).
    Most common normalization for text processing.
    """

    def __init__(out self):
        """Create an NFC normalizer."""
        pass

    def normalize(self, text: String) -> String:
        """Apply NFC normalization."""
        # For now, pass through - full unicode normalization
        # would require unicode tables
        return text


struct NFKCNormalizer(Normalizer):
    """
    NFKC (Compatibility Decomposition, followed by Canonical Composition) normalizer.

    More aggressive than NFC - also normalizes compatibility characters
    (e.g., ﬁ → fi, ² → 2).
    """

    def __init__(out self):
        """Create an NFKC normalizer."""
        pass

    def normalize(self, text: String) -> String:
        """Apply NFKC normalization."""
        # For now, pass through
        return text


struct LowercaseNormalizer(Normalizer):
    """
    Lowercase normalizer.

    Converts all characters to lowercase. Used for case-insensitive models.
    """

    def __init__(out self):
        """Create a lowercase normalizer."""
        pass

    def normalize(self, text: String) -> String:
        """Convert text to lowercase."""
        var result = String()
        for i in range(text.byte_length()):
            var c = text[byte=i]
            var code = ord(c)
            # ASCII uppercase A-Z
            if code >= 65 and code <= 90:
                result += chr(code + 32)
            else:
                result += c
        return result


struct WhitespaceNormalizer(Normalizer):
    """
    Whitespace normalizer.

    Normalizes various whitespace characters to standard space,
    and optionally collapses multiple spaces.
    """

    var collapse_multiple: Bool
    """Whether to collapse multiple spaces into one."""

    def __init__(out self, collapse_multiple: Bool = True):
        """Create a whitespace normalizer."""
        self.collapse_multiple = collapse_multiple

    def normalize(self, text: String) -> String:
        """Normalize whitespace in text."""
        var result = String()
        var prev_was_space = False

        for i in range(text.byte_length()):
            var c = text[byte=i]
            var code = ord(c)

            # Check for various whitespace characters
            var is_whitespace = (code == 32 or  # space
                                 code == 9 or   # tab
                                 code == 10 or  # newline
                                 code == 13 or  # carriage return
                                 code == 160)   # non-breaking space

            if is_whitespace:
                if not self.collapse_multiple or not prev_was_space:
                    result += " "
                prev_was_space = True
            else:
                result += c
                prev_was_space = False

        return result


struct StripNormalizer(Normalizer):
    """
    Strip normalizer.

    Removes leading and/or trailing whitespace.
    """

    var strip_left: Bool
    var strip_right: Bool

    def __init__(out self, strip_left: Bool = True, strip_right: Bool = True):
        """Create a strip normalizer."""
        self.strip_left = strip_left
        self.strip_right = strip_right

    def normalize(self, text: String) -> String:
        """Strip whitespace from text."""
        var start = 0
        var end = text.byte_length()

        if self.strip_left:
            while start < end:
                var code = ord(text[byte=start])
                if code == 32 or code == 9 or code == 10 or code == 13:
                    start += 1
                else:
                    break

        if self.strip_right:
            while end > start:
                var code = ord(text[byte=end - 1])
                if code == 32 or code == 9 or code == 10 or code == 13:
                    end -= 1
                else:
                    break

        return String(text[byte=start:end])


struct ReplaceNormalizer(Normalizer):
    """
    Replace normalizer.

    Replaces a pattern with a replacement string.
    """

    var pattern: String
    var replacement: String

    def __init__(out self, pattern: String, replacement: String):
        """Create a replace normalizer."""
        self.pattern = pattern
        self.replacement = replacement

    def normalize(self, text: String) -> String:
        """Replace pattern in text."""
        if self.pattern.byte_length() == 0:
            return text

        var result = String()
        var i = 0

        while i < text.byte_length():
            # Check for pattern match
            var is_match = True
            if i + self.pattern.byte_length() <= text.byte_length():
                for j in range(self.pattern.byte_length()):
                    if text[byte=i + j] != self.pattern[byte=j]:
                        is_match = False
                        break
            else:
                is_match = False

            if is_match:
                result += self.replacement
                i += self.pattern.byte_length()
            else:
                result += text[byte=i]
                i += 1

        return result


struct NormalizerSequence:
    """
    Sequence of normalizers applied in order.

    Allows chaining multiple normalizations.
    """

    var _normalizers: List[String]  # Normalizer types (simplified)
    var _configs: List[String]      # Serialized configs

    def __init__(out self):
        """Create an empty normalizer sequence."""
        self._normalizers = List[String]()
        self._configs = List[String]()

    def add_lowercase(mut self):
        """Add a lowercase normalizer."""
        self._normalizers.append("lowercase")
        self._configs.append("")

    def add_strip(mut self, left: Bool = True, right: Bool = True):
        """Add a strip normalizer."""
        self._normalizers.append("strip")
        var config = ""
        if left:
            config += "l"
        if right:
            config += "r"
        self._configs.append(config)

    def add_whitespace(mut self, collapse: Bool = True):
        """Add a whitespace normalizer."""
        self._normalizers.append("whitespace")
        self._configs.append("c" if collapse else "")

    def normalize(self, text: String) -> String:
        """Apply all normalizers in sequence."""
        var result = text

        for i in range(len(self._normalizers)):
            var norm_type = self._normalizers[i]
            var config = self._configs[i]

            if norm_type == "lowercase":
                var norm = LowercaseNormalizer()
                result = norm.normalize(result)
            elif norm_type == "strip":
                var left = "l" in config or config.byte_length() == 0
                var right = "r" in config or config.byte_length() == 0
                var norm = StripNormalizer(left, right)
                result = norm.normalize(result)
            elif norm_type == "whitespace":
                var collapse = "c" in config
                var norm = WhitespaceNormalizer(collapse)
                result = norm.normalize(result)

        return result
