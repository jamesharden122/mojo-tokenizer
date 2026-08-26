"""Configuration contract for future model-input construction."""


struct ModelInputConfig(Copyable, Movable):
    """Model sequence policy without model-specific execution behavior."""

    var max_length: Int
    var padding_side: String
    var truncation_side: String
    var pad_token_id: Int64
    var bos_token_id: Optional[Int64]
    var eos_token_id: Optional[Int64]

    def __init__(
        out self,
        max_length: Int,
        pad_token_id: Int64,
        padding_side: String = "right",
        truncation_side: String = "right",
    ):
        self.max_length = max_length
        self.padding_side = padding_side
        self.truncation_side = truncation_side
        self.pad_token_id = pad_token_id
        self.bos_token_id = Optional[Int64]()
        self.eos_token_id = Optional[Int64]()

    def set_bos_token(mut self, token_id: Int64):
        self.bos_token_id = Optional[Int64](token_id)

    def set_eos_token(mut self, token_id: Int64):
        self.eos_token_id = Optional[Int64](token_id)
