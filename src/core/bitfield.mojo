"""Allocation-backed reachability bit field for backtracking BPE."""

# Purpose
# -------
# Store one reachability flag per input-byte position while backtracking BPE
# tokenization. A UInt64 stores 64 flags, so this uses roughly one bit per
# position instead of one byte or more per Bool.
#
# The CPU implementation owns host Allocation[UInt64] storage and provides
# set, clear, successor, and predecessor operations. BitFieldGpuOps is only a
# future interface contract: no device allocation, kernel, or transfer is
# implemented here.

from std.memory.alloc import Allocation, Layout, alloc, dealloc


trait BitFieldCpuOps:
    """Synchronous host operations over a bit field."""

    def is_set(self, bit: Int) -> Bool:
        ...

    def clear(mut self, bit: Int):
        ...

    def set(mut self, bit: Int):
        ...

    def successor(self, bit: Int) -> Int:
        ...

    def predecessor(self, bit: Int) -> Int:
        ...


trait BitFieldGpuOps:
    """Future device-operation boundary; no GPU implementation is provided."""

    def upload(mut self) raises:
        ...

    def clear_gpu(mut self, bit: Int) raises:
        ...

    def set_gpu(mut self, bit: Int) raises:
        ...


struct BitField(BitFieldCpuOps, Movable):
    """A CPU bit field with allocation-owned UInt64 word storage."""

    var words: Allocation[UInt64]
    var _num_bits: Int

    def __init__(out self, bits: Int):
        assert bits > 0, "BitField requires at least one bit"
        self._num_bits = bits
        self.words = alloc(Layout[UInt64](count=(bits + 63) // 64))
        var ptr = self.words.unsafe_ptr()
        for index in range(self.words.layout().count()):
            ptr.unsafe_offset(index).unsafe_write(~UInt64(0))

    def __deinit__(deinit self):
        dealloc(self.words^)

    def is_set(self, bit: Int) -> Bool:
        self._check_bit(bit)
        var words = self.words.unsafe_span()
        return (words[bit // 64] & (UInt64(1) << UInt64(bit % 64))) != 0

    def clear(mut self, bit: Int):
        self._check_bit(bit)
        var words = self.words.unsafe_span()
        words[bit // 64] &= ~(UInt64(1) << UInt64(bit % 64))

    def set(mut self, bit: Int):
        self._check_bit(bit)
        var words = self.words.unsafe_span()
        words[bit // 64] |= UInt64(1) << UInt64(bit % 64)

    def reset_all(mut self):
        """Restore every allocated reachability bit for scratch-buffer reuse."""
        var words = self.words.unsafe_span()
        for index in range(self.words.layout().count()):
            words[index] = ~UInt64(0)

    def successor(self, bit: Int) -> Int:
        """Return the next set bit at or after ``bit``, or ``num_bits``."""

        if bit < 0 or bit >= self._num_bits:
            return self._num_bits
        var word_index = bit // 64
        var word = self.words.unsafe_span()[word_index] >> UInt64(bit % 64)
        if word != 0:
            return bit + _trailing_zeros(word)

        word_index += 1
        var words = self.words.unsafe_span()
        while word_index < self.words.layout().count():
            word = words[word_index]
            if word != 0:
                var result = word_index * 64 + _trailing_zeros(word)
                if result < self._num_bits:
                    return result
                break
            word_index += 1
        return self._num_bits

    def predecessor(self, bit: Int) -> Int:
        """Return the prior set bit at or before ``bit``, or ``-1``."""

        if bit < 0:
            return -1
        var pos = min(bit, self._num_bits - 1)
        var word_index = pos // 64
        var word = self.words.unsafe_span()[word_index] << UInt64(63 - pos % 64)
        if word != 0:
            return pos - _leading_zeros(word)

        var words = self.words.unsafe_span()
        while word_index > 0:
            word_index -= 1
            word = words[word_index]
            if word != 0:
                return word_index * 64 + 63 - _leading_zeros(word)
        return -1

    def num_bits(self) -> Int:
        return self._num_bits

    def _check_bit(self, bit: Int):
        assert bit >= 0 and bit < self._num_bits, "BitField index out of bounds"


@always_inline
def _trailing_zeros(x: UInt64) -> Int:
    if x == 0:
        return 64
    var n = 0
    var value = x
    if (value & 0xFFFFFFFF) == 0:
        n += 32
        value >>= 32
    if (value & 0xFFFF) == 0:
        n += 16
        value >>= 16
    if (value & 0xFF) == 0:
        n += 8
        value >>= 8
    if (value & 0xF) == 0:
        n += 4
        value >>= 4
    if (value & 0x3) == 0:
        n += 2
        value >>= 2
    if (value & 0x1) == 0:
        n += 1
    return n


@always_inline
def _leading_zeros(x: UInt64) -> Int:
    if x == 0:
        return 64
    var n = 0
    var value = x
    if (value & 0xFFFFFFFF00000000) == 0:
        n += 32
        value <<= 32
    if (value & 0xFFFF000000000000) == 0:
        n += 16
        value <<= 16
    if (value & 0xFF00000000000000) == 0:
        n += 8
        value <<= 8
    if (value & 0xF000000000000000) == 0:
        n += 4
        value <<= 4
    if (value & 0xC000000000000000) == 0:
        n += 2
        value <<= 2
    if (value & 0x8000000000000000) == 0:
        n += 1
    return n
