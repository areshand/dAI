# Quality evaluation datasets

`quality-smoke-v1.jsonl` is a small, repository-owned suite for verifying the
collection and scoring pipeline. It is deliberately too small and too simple
to support a model-quality claim. The combined gate defaults to at least 100
paired cases, so the smoke suite cannot accidentally qualify a speed result.

Each non-comment JSONL row requires:

- `id`: stable unique case identifier;
- `category`: aggregation label;
- `prompt`: text sent to the completions endpoint;
- `scorer`: an object with `type` and its expected answer; and
- optional `max_tokens`, defaulting to 32.

Supported scorers are `literal`, `normalized_exact`, `choice`, `numeric`, and
`contains_all`. For a claim-bearing run, provide a representative, licensed
suite with at least 100 cases and preserve its exact file hash. Set
`DAI_QUALITY_DATASET_LOCAL` to its local path before starting the AWS runner.

Quality and speed are intentionally collected separately. Quality prompts keep
their natural lengths; the speed gate continues to use the fixed 1,000-token
prompt and 256-token output. This avoids padding benchmark questions with
irrelevant text merely to make their timing shapes identical.

## Standard reasoning slice

`prepare_gsm8k_quality_suite.py` reproducibly downloads the official archived
GSM8K test split at an immutable commit, verifies its SHA-256 digest, and takes
a seeded sample. GSM8K is MIT-licensed; the derived data stays under ignored
`prototype/results/` rather than being copied into this repository.

```bash
python3 prototype/prepare_gsm8k_quality_suite.py \
  --limit 200 \
  --output prototype/results/quality/gsm8k-200.jsonl

DAI_VARIANT_SET=quality \
DAI_QUALITY_DATASET_LOCAL="$PWD/prototype/results/quality/gsm8k-200.jsonl" \
./scripts/aws-generation-experiment.sh
```

Passing this suite supports a scoped arithmetic-reasoning non-inferiority
claim. It does not establish broad chat, coding, factuality, multilingual, or
safety quality; those require additional datasets in the same JSONL schema.
