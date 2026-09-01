# Remaining work

## Model-specific contract

- [ ] Select the first ONNX / `.surml` model.
- [ ] Record its input names, element types, dimensions, and dynamic axes.
- [ ] Define its vocabulary and merge-table provisioning strategy.
- [ ] Record BOS/EOS/CLS/SEP/PAD IDs and padding/truncation policy.
- [ ] Generate local golden vectors from the tokenizer used during model export.
- [ ] Define explicit bindings from standalone tokenizer outputs to embedding
  model inputs; do not rely on coincidentally matching tensor names.

## ONNX integration

- [x] Add fixed-shape allocation-backed batch and single-vector tensor staging.
- [ ] Encode staged token IDs and source text as ONNX `TensorProto` values.
- [ ] Capture tokenizer calls and token IDs as ONNX constant graphs.
- [ ] Preserve source text losslessly as a `UINT8` UTF-8 tensor.
- [ ] Define a reusable standalone tokenizer `ModelProto` contract for arbitrary
  `UINT8` UTF-8 input, distinct from the existing constant capture fixture.
- [ ] Decide whether the reusable tokenizer graph uses standard ONNX operators,
  a versioned custom operator, or a runtime adapter.
- [ ] Add an optional model-policy layer for padding, truncation, and masks.
- [ ] Make the standalone tokenizer model emit every tensor required by its
  declared contract, including masks when model policy enables them.
- [ ] Define a versioned project protobuf bundle containing a tokenizer
  `ModelProto`, an embedding `ModelProto`, and their tensor bindings.
- [ ] Keep tokenizer export independent of bundle and embedding-model export so
  the tokenizer artifact can be distributed and executed on its own.
- [ ] Split `src/onnx_utils/` by responsibility: tokenizer model, model bundle,
  tensor bindings, validation, and generic protobuf wire encoding.
- [ ] Validate bundle bindings for tensor names, element types, dimensions,
  dynamic axes, required masks, and embedding-table vocabulary size.
- [ ] Add round-trip serialization tests for both a standalone tokenizer model
  and the two-model bundle.
- [ ] Validate generated captures with ONNX Runtime in CI.

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
- [ ] Revisit the per-prefix `IntervalTree` promotion policy if Mojo adds an
  allocation-free point-query API; the current benchmark shows flat scans are
  faster across the complete 8-to-256 byte-fanout range.

## Economic semantic-matching pipeline

### Goal and boundaries

- [ ] Define v1 as **evidence retrieval**: given an economics/finance question,
  rank its most relevant text fact, table row, or table cell group.
- [ ] Record the input/output contract: `query_text`, `candidate_text`,
  `candidate_id`, `relevance`, source dataset, and split.
- [ ] Keep this package responsible only for vocabulary, tokenization, and
  model-input tensors; put corpus preparation and neural matching in a sibling
  `econ_matcher` module.
- [ ] Configure the local corpus root with `ECON_MATCH_DATA_ROOT`; do not commit
  raw datasets, generated binary tensors, checkpoints, or absolute paths.
- [ ] Review dataset licenses before combining or distributing derived training
  artifacts (EconLogicQA is marked CC BY-NC-SA 4.0).

### 1. Inventory and normalize the economics corpora

- [ ] Unpack `conv_finqa/data.zip` into a local ignored data cache and record its
  source version/checksum in a dataset manifest.
- [ ] Use FinQA as the first supervised retrieval corpus: question -> gold text
  evidence and annotated table rows/cells.
- [ ] Add ConvFinQA turn-level data after FinQA works; include dialogue history
  only through an explicit, versioned prompt/serialization policy.
- [ ] Treat EconNLI as an auxiliary causal-relation pair-classification task, not
  as direct evidence-retrieval labels.
- [ ] Treat STEER-ME and EconLogicQA as concept/reasoning evaluation or auxiliary
  tasks; do not mix their answer labels with retrieval relevance labels.
- [ ] Use STS-B only as a generic semantic-similarity calibration baseline.
- [ ] Define one canonical, line-oriented record format for all derived examples:
  `id`, `task`, `split`, `query`, `candidate`, `label`, and JSON metadata.
- [ ] Preserve dataset-provided splits and prevent leakage: group FinQA by report,
  ConvFinQA by conversation, and template-generated data by template family.
- [ ] Generate FinQA positives from `qa.gold_inds`; generate initial negatives
  from other facts/rows in the same report, then from other reports.
- [ ] Serialize tables deterministically, for example
  `<row> <cell> year <cell> 2024 <row> <cell> revenue <cell> $1,200`.
- [ ] Preserve raw numbers and add normalized numeric metadata (currency,
  percentage, date, magnitude) so matching does not lose financial meaning.

### 2. Build a reproducible domain BPE vocabulary

- [ ] Connect the Mojo corpus-to-vocabulary trainer to `read_bin` text-table
  chunks once its `StringTableReader` exists; it currently trains from
  caller-provided training-split strings and starts from the complete byte base.
- [ ] Keep a future `read_bin` → Rust FFI separate from the Mojo trainer API.
- [ ] Add a stable vocabulary/merge serialization format and loader under
  `src/io/`, including format version, special-token IDs, corpus manifest hash,
  and normalization policy.
- [ ] Reserve and document non-overlapping IDs for at least `<pad>`, `<unk>`,
  `<bos>`, `<eos>`, `<sep>`, `<query>`, `<candidate>`, `<row>`, and `<cell>`.
- [ ] Decide whether numeric placeholders such as `<num>`, `<percent>`, and
  `<currency>` augment raw values; test the chosen policy on financial tables.
- [ ] Define lowercase/Unicode/whitespace normalization rules and make them
  identical during vocabulary training, offline preprocessing, and inference.
- [ ] Add round-trip and determinism tests for vocabulary loading, merges,
  special-token recognition, unseen UTF-8 text, financial punctuation, and
  numeric/table serialization.
- [ ] Measure token coverage, unknown/fallback rate, and sequence-length
  distribution for every corpus before freezing the vocabulary.

### 3. Produce model-ready token tensors

- [ ] Add a fixed-length ONNX capture policy with padding, truncation,
  attention masks, and optional BOS/EOS tokens.
- [ ] Create separate query and candidate sequence builders rather than encoding
  a pair into one sequence for the first retrieval model.
- [ ] Emit deterministic `Int64` token IDs plus masks to ignored binary artifacts
  consumable by `read_bin` and the neural-network project.
- [ ] Store record IDs, labels, source metadata, and tokenizer-format version
  alongside every tensor shard.
- [ ] Add a small golden fixture covering a question, a text fact, and a table
  row so preprocessing can be verified without local raw datasets.

### 4. Implement the neural matching baseline

- [ ] Create a sibling `econ_matcher` module with `data/`, `model/`, `training/`,
  and `evaluation/` boundaries.
- [ ] Start with a shared-weight bi-encoder: token embedding -> masked mean
  pooling -> dense projection -> L2 normalization.
- [ ] Score query/candidate pairs with cosine similarity; retrieve candidates by
  top-k score.
- [ ] Add an embedding layer, masked pooling operation, cosine scorer, and
  contrastive ranking loss to `neural_network_builder` only if they are generic
  reusable primitives; keep task-specific logic in `econ_matcher`.
- [ ] Train with one positive and in-batch negatives first, then add same-report
  and model-mined hard negatives once the baseline is stable.
- [ ] Version model hyperparameters, tokenizer artifact, data manifest, random
  seed, and checkpoint format for each run.

### 5. Evaluate and iterate

- [ ] Establish lexical baselines (exact token overlap and BM25-style scoring)
  before evaluating neural matching.
- [ ] Report FinQA/ConvFinQA Recall@1, Recall@5, MRR, and nDCG for evidence
  retrieval; separately report text and table evidence results.
- [ ] Report Spearman correlation on STS-B and task-appropriate accuracy/F1 for
  EconNLI, STEER-ME, and EconLogicQA auxiliary evaluations.
- [ ] Slice results by numeric content, table versus prose, question length,
  unseen economic terms, and source dataset.
- [ ] Inspect false positives/negatives and add hard-negative rules only when a
  measured failure mode supports them.
- [ ] Promote a tokenizer/model pair only after it beats the lexical baseline on
  held-out FinQA evidence retrieval without degrading numeric/table slices.

### Milestone order

- [ ] M1: Canonical FinQA records, deterministic table serialization, and a
  frozen train-only tokenizer artifact.
- [ ] M2: Query/candidate tensors and a golden preprocessing test.
- [ ] M3: Bi-encoder baseline that returns ranked FinQA evidence candidates.
- [ ] M4: ConvFinQA support, hard-negative mining, and full evaluation dashboard.
- [ ] M5: Add auxiliary economics tasks only after M3/M4 metrics are reproducible.
