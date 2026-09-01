"""Token IDs aligned to half-open UTF-8 byte intervals."""

from std.collections.interval import Interval


struct TokenSpan(Copyable, Movable):
    """One emitted token and its ``[start, end)`` input-byte span."""

    var token_id: Int
    var span: Interval[Int]

    def __init__(out self, token_id: Int, span: Interval[Int]):
        self.token_id = token_id
        self.span = span

    def shifted(self, byte_offset: Int) -> Self:
        return Self(
            self.token_id,
            Interval(self.span.start + byte_offset, self.span.end + byte_offset),
        )
