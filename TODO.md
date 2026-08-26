# Remaining work

## Model-specific contract

- [ ] Select the first ONNX / `.surml` model.
- [ ] Record its input names, element types, dimensions, and dynamic axes.
- [ ] Define its vocabulary and merge-table provisioning strategy.
- [ ] Record BOS/EOS/CLS/SEP/PAD IDs and padding/truncation policy.
- [ ] Generate local golden vectors from the tokenizer used during model export.

## Model-input implementation

- [ ] Implement a concrete `ModelInputBuilder` for the selected model.
- [ ] Add special tokens, truncate, pad, and generate required masks.
- [ ] Validate tensor shape and flattened-value length.
- [ ] Add any required token-type or position-ID tensors.
- [ ] Keep all emitted token buffers as `Int64`.

## SurrealML integration

- [ ] Add named, typed, multidimensional tensor inputs to SurrealML.
- [ ] Discover and validate ONNX input specifications at model load time.
- [ ] Convert tensors to matching ONNX Runtime values without passing through
  `f32`.
- [ ] Preserve the existing single-`f32` API as a compatibility wrapper in
  SurrealML, not in this tokenizer package.

## Validation

- [ ] Compare exact token IDs and every model input tensor with golden vectors.
- [ ] Compare model output against the original ONNX export environment.
- [ ] Profile cache and backtracking behavior before further optimization.
