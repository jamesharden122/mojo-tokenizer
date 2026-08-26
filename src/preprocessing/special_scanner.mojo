"""
SIMD-optimized special-token boundary scanning.

Fast detection of special characters and token boundaries
using 16-byte SIMD chunks.
"""

comptime SIMD_WIDTH: Int = 16


@always_inline
def char_matches_any(c: UInt8, targets: List[UInt8]) -> Bool:
    """Check if character matches any target."""
    for i in range(targets.byte_length()):
        if c == targets[i]:
            return True
    return False


def find_char_simd(data: String, start: Int, target: UInt8) -> Int:
    """
    Find first occurrence of target character using SIMD.

    Returns position of target, or -1 if not found.
    """
    var pos = start
    var n = data.byte_length()

    # SIMD processing
    while pos + SIMD_WIDTH <= n:
        var chunk = SIMD[DType.uint8, SIMD_WIDTH]()

        comptime
        for i in range(SIMD_WIDTH):
            chunk[i] = UInt8(ord(data[byte=pos + i]))

        # Create match mask
        var match_mask = SIMD[DType.uint8, SIMD_WIDTH]()
        comptime
        for i in range(SIMD_WIDTH):
            match_mask[i] = 1 if chunk[i] == target else 0

        # Any matches?
        if match_mask.reduce_add() > 0:
            comptime
            for i in range(SIMD_WIDTH):
                if match_mask[i] == 1:
                    return pos + i

        pos += SIMD_WIDTH

    # Scalar tail
    while pos < n:
        if UInt8(ord(data[byte=pos])) == target:
            return pos
        pos += 1

    return -1


def find_any_char_simd(data: String, start: Int, targets: String) -> Int:
    """
    Find first occurrence of any character in targets using SIMD.

    Returns position of first match, or -1 if not found.
    """
    if targets.byte_length() == 0:
        return -1

    var pos = start
    var n = data.byte_length()

    # Pre-convert targets to bytes
    var target_bytes = List[UInt8]()
    for i in range(targets.byte_length()):
        target_bytes.append(UInt8(ord(targets[byte=i])))

    # SIMD processing
    while pos + SIMD_WIDTH <= n:
        var chunk = SIMD[DType.uint8, SIMD_WIDTH]()

        comptime
        for i in range(SIMD_WIDTH):
            chunk[i] = UInt8(ord(data[byte=pos + i]))

        # Check each target
        var found_mask = SIMD[DType.uint8, SIMD_WIDTH](0)
        for ti in range(len(target_bytes)):
            var t = target_bytes[ti]
            comptime
            for i in range(SIMD_WIDTH):
                if chunk[i] == t:
                    found_mask[i] = 1

        # Any matches?
        if found_mask.reduce_add() > 0:
            comptime
            for i in range(SIMD_WIDTH):
                if found_mask[i] == 1:
                    return pos + i

        pos += SIMD_WIDTH

    # Scalar tail
    while pos < n:
        var c = UInt8(ord(data[byte=pos]))
        for ti in range(len(target_bytes)):
            if c == target_bytes[ti]:
                return pos
        pos += 1

    return -1


def find_special_token_boundary(data: String, start: Int, special_start: String) -> Int:
    """
    Find start of a special token marker (e.g., "<|" or "[").

    Used for quickly locating special tokens during tokenization.
    Returns position of marker, or -1 if not found.
    """
    if special_start.byte_length() == 0:
        return -1

    var first_char = UInt8(ord(special_start[byte=0]))
    var pos = start

    while pos < data.byte_length():
        # Find first character
        var found = find_char_simd(data, pos, first_char)
        if found < 0:
            return -1

        # Check if full marker matches
        if found + special_start.byte_length() <= data.byte_length():
            var matches = True
            for i in range(special_start.byte_length()):
                if data[byte=found + i] != special_start[byte=i]:
                    matches = False
                    break
            if matches:
                return found

        pos = found + 1

    return -1


def count_char_simd(data: String, target: UInt8) -> Int:
    """
    Count occurrences of target character using SIMD.

    Returns total count.
    """
    var count = 0
    var pos = 0
    var n = data.byte_length()

    # SIMD processing
    while pos + SIMD_WIDTH <= n:
        var chunk = SIMD[DType.uint8, SIMD_WIDTH]()

        comptime
        for i in range(SIMD_WIDTH):
            chunk[i] = UInt8(ord(data[byte=pos + i]))

        # Create match mask and count
        var match_mask = SIMD[DType.uint8, SIMD_WIDTH]()
        comptime
        for i in range(SIMD_WIDTH):
            match_mask[i] = 1 if chunk[i] == target else 0

        count += Int(match_mask.reduce_add())
        pos += SIMD_WIDTH

    # Scalar tail
    while pos < n:
        if UInt8(ord(data[byte=pos])) == target:
            count += 1
        pos += 1

    return count
