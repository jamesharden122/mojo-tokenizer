"""Typed, named buffers for a future ONNX/SurrealML boundary."""

from std.memory import unsafe_destroy_n
from std.memory.alloc import Allocation, Layout, alloc, dealloc


def allocate_int64_buffer(value_count: Int, fill: Int64 = 0) -> Allocation[Int64]:
    """Allocate initialized Int64 storage for a model-input tensor."""

    assert value_count >= 0, "Int64 buffer size cannot be negative"
    var result = alloc(Layout[Int64](count=value_count))
    var pointer = result.unsafe_ptr()
    for index in range(value_count):
        pointer.unsafe_offset(index).unsafe_write(fill)
    return result^


def copy_int64_buffer(source: Allocation[Int64]) -> Allocation[Int64]:
    """Return an independent copy of an Int64 allocation."""

    var result = allocate_int64_buffer(source.layout().count())
    var source_values = source.unsafe_span()
    var result_values = result.unsafe_span()
    for index in range(source.layout().count()):
        result_values[index] = source_values[index]
    return result^


def destroy_int64_buffer(var buffer: Allocation[Int64]):
    """Release initialized Int64 storage."""

    unsafe_destroy_n(buffer.unsafe_ptr(), buffer.layout().count())
    dealloc(buffer^)


struct NamedInt64Tensor(Movable):
    """A flattened Int64 tensor with an explicit name and shape."""

    var name: String
    var shape: List[Int]
    var values: Allocation[Int64]

    def __init__(
        out self, name: String, shape: List[Int], var values: Allocation[Int64]
    ):
        self.name = name
        self.shape = shape.copy()
        self.values = values^

    def __deinit__(deinit self):
        destroy_int64_buffer(self.values^)


struct TokenizedInputs(Movable):
    """Named integer tensors commonly consumed by text ONNX models."""

    var input_ids: NamedInt64Tensor
    var attention_mask: Optional[NamedInt64Tensor]
    var token_type_ids: Optional[NamedInt64Tensor]
    var position_ids: Optional[NamedInt64Tensor]

    def __init__(out self, var input_ids: NamedInt64Tensor):
        self.input_ids = input_ids^
        self.attention_mask = Optional[NamedInt64Tensor]()
        self.token_type_ids = Optional[NamedInt64Tensor]()
        self.position_ids = Optional[NamedInt64Tensor]()

    def set_attention_mask(mut self, var tensor: NamedInt64Tensor):
        self.attention_mask = Optional[NamedInt64Tensor](tensor^)

    def set_token_type_ids(mut self, var tensor: NamedInt64Tensor):
        self.token_type_ids = Optional[NamedInt64Tensor](tensor^)

    def set_position_ids(mut self, var tensor: NamedInt64Tensor):
        self.position_ids = Optional[NamedInt64Tensor](tensor^)
