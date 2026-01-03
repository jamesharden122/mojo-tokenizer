"""
Post-processors for tokenization pipeline.

Post-processors transform token sequences after tokenization:
- Add special tokens (BOS, EOS, CLS, SEP)
- Apply templates for single/pair sequences
- Handle padding and truncation
"""


struct EncodingOutput(Copyable, Movable):
    """Output from tokenization with metadata."""

    var ids: List[Int]
    """Token IDs."""

    var type_ids: List[Int]
    """Token type IDs (for BERT-style models)."""

    var attention_mask: List[Int]
    """Attention mask (1 for real tokens, 0 for padding)."""

    var special_tokens_mask: List[Int]
    """Mask for special tokens (1 for special, 0 for regular)."""

    var offsets: List[Tuple[Int, Int]]
    """Character offsets for each token (start, end)."""

    fn __init__(out self):
        """Create empty encoding output."""
        self.ids = List[Int]()
        self.type_ids = List[Int]()
        self.attention_mask = List[Int]()
        self.special_tokens_mask = List[Int]()
        self.offsets = List[Tuple[Int, Int]]()

    fn __copyinit__(out self, existing: Self):
        """Copy constructor."""
        self.ids = existing.ids.copy()
        self.type_ids = existing.type_ids.copy()
        self.attention_mask = existing.attention_mask.copy()
        self.special_tokens_mask = existing.special_tokens_mask.copy()
        self.offsets = existing.offsets.copy()

    fn __moveinit__(out self, deinit existing: Self):
        """Move constructor."""
        self.ids = existing.ids^
        self.type_ids = existing.type_ids^
        self.attention_mask = existing.attention_mask^
        self.special_tokens_mask = existing.special_tokens_mask^
        self.offsets = existing.offsets^

    fn add_token(mut self, id: Int, type_id: Int = 0, is_special: Bool = False):
        """Add a token to the encoding."""
        self.ids.append(id)
        self.type_ids.append(type_id)
        self.attention_mask.append(1)
        self.special_tokens_mask.append(1 if is_special else 0)


trait PostProcessor:
    """Base trait for post-processors."""

    fn process(self, encoding: EncodingOutput, add_special_tokens: Bool) -> EncodingOutput:
        """Process a single sequence encoding."""
        ...

    fn process_pair(
        self,
        encoding: EncodingOutput,
        pair_encoding: EncodingOutput,
        add_special_tokens: Bool
    ) -> EncodingOutput:
        """Process a pair of sequence encodings (for sentence pairs)."""
        ...


struct BertPostProcessor(PostProcessor):
    """
    BERT-style post-processor.

    Adds [CLS] at start and [SEP] at end.
    For pairs: [CLS] A [SEP] B [SEP]
    """

    var cls_id: Int
    var sep_id: Int

    fn __init__(out self, cls_id: Int = 101, sep_id: Int = 102):
        """Create a BERT post-processor."""
        self.cls_id = cls_id
        self.sep_id = sep_id

    fn process(self, encoding: EncodingOutput, add_special_tokens: Bool) -> EncodingOutput:
        """Process single sequence: [CLS] tokens [SEP]."""
        var result = EncodingOutput()

        if add_special_tokens:
            result.add_token(self.cls_id, 0, True)

        for i in range(len(encoding.ids)):
            result.add_token(
                encoding.ids[i],
                0,
                encoding.special_tokens_mask[i] == 1
            )

        if add_special_tokens:
            result.add_token(self.sep_id, 0, True)

        return result^

    fn process_pair(
        self,
        encoding: EncodingOutput,
        pair_encoding: EncodingOutput,
        add_special_tokens: Bool
    ) -> EncodingOutput:
        """Process pair: [CLS] A [SEP] B [SEP]."""
        var result = EncodingOutput()

        if add_special_tokens:
            result.add_token(self.cls_id, 0, True)

        # First sequence (type_id = 0)
        for i in range(len(encoding.ids)):
            result.add_token(
                encoding.ids[i],
                0,
                encoding.special_tokens_mask[i] == 1
            )

        if add_special_tokens:
            result.add_token(self.sep_id, 0, True)

        # Second sequence (type_id = 1)
        for i in range(len(pair_encoding.ids)):
            result.add_token(
                pair_encoding.ids[i],
                1,
                pair_encoding.special_tokens_mask[i] == 1
            )

        if add_special_tokens:
            result.add_token(self.sep_id, 1, True)

        return result^


struct TemplatePostProcessor(PostProcessor):
    """
    Template-based post-processor.

    Uses templates to define how tokens are combined.
    Template format: "$A" for first sequence, "$B" for second, literals for special tokens.

    Example templates:
        single: "[CLS] $A [SEP]"
        pair: "[CLS] $A [SEP] $B [SEP]"
    """

    var single_template: String
    var pair_template: String
    var special_tokens: Dict[String, Int]

    fn __init__(out self, single: String, pair: String):
        """Create a template post-processor."""
        self.single_template = single
        self.pair_template = pair
        self.special_tokens = Dict[String, Int]()

    fn add_special_token(mut self, text: String, id: Int):
        """Register a special token for template expansion."""
        self.special_tokens[text] = id

    fn process(self, encoding: EncodingOutput, add_special_tokens: Bool) -> EncodingOutput:
        """Process using single template."""
        return self._apply_template(self.single_template, encoding, EncodingOutput(), add_special_tokens)

    fn process_pair(
        self,
        encoding: EncodingOutput,
        pair_encoding: EncodingOutput,
        add_special_tokens: Bool
    ) -> EncodingOutput:
        """Process using pair template."""
        return self._apply_template(self.pair_template, encoding, pair_encoding, add_special_tokens)

    fn _apply_template(
        self,
        template: String,
        a: EncodingOutput,
        b: EncodingOutput,
        add_special: Bool
    ) -> EncodingOutput:
        """Apply template to encodings."""
        var result = EncodingOutput()
        var i = 0
        var n = len(template)

        while i < n:
            var ch = String(template[i])
            if ch == "$" and i + 1 < n:
                var next_ch = String(template[i + 1])
                if next_ch == "A":
                    # Insert sequence A
                    for j in range(len(a.ids)):
                        result.add_token(a.ids[j], 0, a.special_tokens_mask[j] == 1)
                    i += 2
                elif next_ch == "B":
                    # Insert sequence B
                    for j in range(len(b.ids)):
                        result.add_token(b.ids[j], 1, b.special_tokens_mask[j] == 1)
                    i += 2
                else:
                    i += 1
            elif ch == "[":
                # Special token - find closing bracket
                var token_start = i
                while i < n and String(template[i]) != "]":
                    i += 1
                if i < n:
                    i += 1  # Skip ]
                var token_text = String(template[token_start:i])

                if add_special and token_text in self.special_tokens:
                    try:
                        result.add_token(self.special_tokens[token_text], 0, True)
                    except:
                        pass
            elif ch == " ":
                # Skip whitespace in template
                i += 1
            else:
                i += 1

        return result^


struct GPT2PostProcessor(PostProcessor):
    """
    GPT-2 style post-processor.

    Optionally adds BOS/EOS tokens. GPT-2 originally used no special tokens
    during training, but <|endoftext|> is commonly used.
    """

    var bos_id: Int
    var eos_id: Int
    var add_bos: Bool
    var add_eos: Bool

    fn __init__(
        out self,
        bos_id: Int = -1,
        eos_id: Int = 50256,  # <|endoftext|>
        add_bos: Bool = False,
        add_eos: Bool = True
    ):
        """Create a GPT-2 post-processor."""
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.add_bos = add_bos
        self.add_eos = add_eos

    fn process(self, encoding: EncodingOutput, add_special_tokens: Bool) -> EncodingOutput:
        """Process single sequence."""
        var result = EncodingOutput()

        if add_special_tokens and self.add_bos and self.bos_id >= 0:
            result.add_token(self.bos_id, 0, True)

        for i in range(len(encoding.ids)):
            result.add_token(
                encoding.ids[i],
                0,
                encoding.special_tokens_mask[i] == 1
            )

        if add_special_tokens and self.add_eos and self.eos_id >= 0:
            result.add_token(self.eos_id, 0, True)

        return result^

    fn process_pair(
        self,
        encoding: EncodingOutput,
        pair_encoding: EncodingOutput,
        add_special_tokens: Bool
    ) -> EncodingOutput:
        """Process pair (concatenate with optional EOS between)."""
        var result = EncodingOutput()

        if add_special_tokens and self.add_bos and self.bos_id >= 0:
            result.add_token(self.bos_id, 0, True)

        # First sequence
        for i in range(len(encoding.ids)):
            result.add_token(
                encoding.ids[i],
                0,
                encoding.special_tokens_mask[i] == 1
            )

        # Optionally add separator
        if add_special_tokens and self.add_eos and self.eos_id >= 0:
            result.add_token(self.eos_id, 0, True)

        # Second sequence
        for i in range(len(pair_encoding.ids)):
            result.add_token(
                pair_encoding.ids[i],
                0,
                pair_encoding.special_tokens_mask[i] == 1
            )

        if add_special_tokens and self.add_eos and self.eos_id >= 0:
            result.add_token(self.eos_id, 0, True)

        return result^


struct LlamaPostProcessor(PostProcessor):
    """
    Llama-style post-processor.

    Adds <s> (BOS) at start. EOS (</s>) typically added during generation.
    """

    var bos_id: Int
    var eos_id: Int

    fn __init__(out self, bos_id: Int = 1, eos_id: Int = 2):
        """Create a Llama post-processor."""
        self.bos_id = bos_id
        self.eos_id = eos_id

    fn process(self, encoding: EncodingOutput, add_special_tokens: Bool) -> EncodingOutput:
        """Process single sequence: <s> tokens."""
        var result = EncodingOutput()

        if add_special_tokens:
            result.add_token(self.bos_id, 0, True)

        for i in range(len(encoding.ids)):
            result.add_token(
                encoding.ids[i],
                0,
                encoding.special_tokens_mask[i] == 1
            )

        return result^

    fn process_pair(
        self,
        encoding: EncodingOutput,
        pair_encoding: EncodingOutput,
        add_special_tokens: Bool
    ) -> EncodingOutput:
        """Process pair: <s> A </s> <s> B."""
        var result = EncodingOutput()

        if add_special_tokens:
            result.add_token(self.bos_id, 0, True)

        for i in range(len(encoding.ids)):
            result.add_token(encoding.ids[i], 0, encoding.special_tokens_mask[i] == 1)

        if add_special_tokens:
            result.add_token(self.eos_id, 0, True)
            result.add_token(self.bos_id, 0, True)

        for i in range(len(pair_encoding.ids)):
            result.add_token(pair_encoding.ids[i], 0, pair_encoding.special_tokens_mask[i] == 1)

        return result^
