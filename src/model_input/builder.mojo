"""Interface for future model-specific token-buffer builders."""

from .buffers import TokenizedInputs


trait ModelInputBuilder:
    """Build typed model inputs from text.

    Concrete padding, truncation, mask generation, and model policy are
    intentionally deferred.
    """

    def build(mut self, text: String) raises -> TokenizedInputs:
        ...
