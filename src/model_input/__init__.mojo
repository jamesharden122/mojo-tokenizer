"""Public model-input contracts; construction behavior is not implemented."""

from .config import ModelInputConfig
from .buffers import (
    NamedInt64Tensor,
    TokenizedInputs,
    allocate_int64_buffer,
    copy_int64_buffer,
    destroy_int64_buffer,
)
from .builder import ModelInputBuilder
