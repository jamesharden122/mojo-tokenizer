from layout import Layout, LayoutTensor
from layout.layout_tensor import LayoutTensorIter
from max.algorithm import parallelize
from std.algorithm import map
from std.collections.span import Span
from std.memory import Pointer
from std.memory.alloc import Allocation, Layout as AllLayout, alloc

# Create tensor on CPU using Array to allocate storage space.


trait ToOnnx:
    @staticmethod
    def to_onnx_model() raises:
        ...

    @staticmethod
    def to_onnx_graph() raises:
        ...

    @staticmethod
    def to_onnx_tensor() raises:
        ...


trait Tens2D:
    """Parallel row operations for contiguous row-major tensors."""

    @staticmethod
    def _fill_row[
        W: Int, T: DType, O1: MutOrigin
    ](dst_ptr: Pointer[Scalar[T], O1], row: Int, vals: InlineArray[Scalar[T], W],):
        """Fill ``row`` with ``vals`` in parallel."""

        @parameter
        def fill_one_row(col: Int):
            dst_ptr.unsafe_store(row * W + col, vals[col])

        map[fill_one_row](W)

    @staticmethod
    def _append_tensors[
        W: Int,
        DH: Int,
        SH: Int,
        T: DType,
        O2: MutOrigin,
        O3: Origin,
    ](dst_ptr: Pointer[Scalar[T], O2], rows_ptr: Pointer[Scalar[T], O3],):
        """Append ``SH`` source rows after ``DH`` destination rows in parallel."""

        @parameter
        def copy_one_row(row: Int):
            for col in range(W):
                var source_index = row * W + col
                var destination_index = (DH + row) * W + col
                dst_ptr.unsafe_store(destination_index, rows_ptr.unsafe_load(source_index))

        map[copy_one_row](SH)


struct OnnxTens[SR: Int, BS: Int, NumCols: Int, dtype: DType, O: Origin](Tens2D):
    comptime RowLyt = Layout.row_major(1, Self.NumCols)
    comptime BatchLyt = Layout.row_major(Self.BS, Self.NumCols)
    comptime OutLyt = Layout.row_major(Self.SR, Self.NumCols)
    comptime BatchElem = Self.BS * Self.NumCols
    comptime BatchCount = Self.SR // Self.BS
    comptime SourceElem = Self.SR * Self.NumCols

    comptime RowView = LayoutTensor[Self.dtype, Self.RowLyt, Self.O]
    comptime ValView = LayoutTensor[Self.dtype, Self.BatchLyt, Self.O]
    comptime SrcDataIter = LayoutTensorIter[Self.dtype, Self.BatchLyt, Self.O]
    var val_view: Self.ValView
    var row_view: Self.RowView
    var batch_not_row: Bool

    def __init__(
        out self,
        values: Span[Scalar[Self.dtype], Self.O],
        batch_not_row: Bool = True,
    ):
        comptime assert Self.SR > 0, "source row count must be positive"
        comptime assert Self.BS > 0, "batch size must be positive"
        comptime assert Self.NumCols > 0, "column count must be positive"
        comptime assert Self.SR % Self.BS == 0, "source rows must divide evenly into batches"
        assert len(values) >= Self.SourceElem, "source span is smaller than the declared tensor"
        # Both are zero-copy views; the mode selects which writer may use them.
        self.val_view = Self.ValView(values)
        self.row_view = Self.RowView(values)
        self.batch_not_row = batch_not_row

    def _src_data_iter[
        OO: MutOrigin
    ](self, out_ptr: Pointer[Scalar[Self.dtype], OO], C: Int,):
        """Copy the source batch beginning at row ``C`` into the output tensor."""

        var bound = Self.SrcDataIter.linear_uint_type(Self.SourceElem)
        var stride = Self.SrcDataIter.linear_uint_type(Self.BatchElem)
        var offset = Self.SrcDataIter.linear_uint_type(C * Self.NumCols)
        # Each worker creates its own iterator positioned at source row C.
        var src_data_iter = Self.SrcDataIter(
            self.val_view.ptr,
            bound,
            stride=stride,
            offset=offset,
        )
        var right_tensor = src_data_iter[]

        for row in range(Self.BS):
            for col in range(Self.NumCols):
                # C is also the first destination row owned by this batch.
                var destination_index = (C + row) * Self.NumCols + col
                out_ptr.unsafe_offset(destination_index).unsafe_write(right_tensor[row, col][0])

    def write_tensor(self) -> Allocation[Scalar[Self.dtype]]:
        """Copy all source batches into a contiguous output tensor in parallel."""
        assert self.batch_not_row, "write_tensor requires batch mode"
        var output = alloc(AllLayout[Scalar[Self.dtype]](count=Self.SourceElem))
        var out_tensor = LayoutTensor[Self.dtype, Self.OutLyt](output.unsafe_ptr())
        var out_ptr = out_tensor.ptr

        @parameter
        def write_batch(batch_index: Int):
            # Batch ranges are disjoint, so the parallel writes need no lock.
            var C = batch_index * Self.BS
            self._src_data_iter(out_ptr, C)

        parallelize[write_batch](Self.BatchCount)
        # Ownership transfers to the caller, which must eventually deallocate it.
        return output^

    def write_vector(self) -> Allocation[Scalar[Self.dtype]]:
        """Copy one source row into a contiguous ONNX vector allocation."""

        comptime assert Self.SR == 1, "write_vector requires one source row"
        comptime assert Self.BS == 1, "write_vector requires a batch size of one"
        assert not self.batch_not_row, "write_vector requires row mode"

        var output = alloc(AllLayout[Scalar[Self.dtype]](count=Self.NumCols))
        var out_tensor = LayoutTensor[Self.dtype, Self.RowLyt](output.unsafe_ptr())
        var out_ptr = out_tensor.ptr
        for col in range(Self.NumCols):
            out_ptr.unsafe_offset(col).unsafe_write(self.row_view[0, col][0])

        # Ownership transfers to the request handler.
        return output^
