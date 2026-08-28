# AWS generation speed experiment — 2026-08-22

## Outcome

The tested SGLang speculative-decoding configurations did **not** produce a
correctness-qualified 10× speedup for Qwen3-30B-A3B. The largest raw result was
5.49× median decode throughput from n-gram speculation, but its ten measured
runs produced two output hashes and neither matched the baseline hash. It is a
performance observation, not a valid same-output speedup.

## Method

- AWS profile: `mi:scratchpad`
- Region and instance: `us-east-2`, one `g7e.4xlarge` (NVIDIA RTX PRO 6000
  Blackwell, 96 GB)
- Model: pinned local/S3 copy of Qwen3-30B-A3B in BF16
- Server: pinned `lmsysorg/sglang:v0.5.16`
- Workload: batch one, first exactly 1,000 tokens of the project design as the
  prompt, exactly 256 generated tokens
- Sampling/repetition: fixed server seed 1234, deterministic inference, Triton
  attention, two warmups, ten measured repetitions per variant
- Metrics: streaming time to first token (TTFT), end-to-end latency, and decode
  output tokens per second
- Correctness gate: one output hash within a variant and exact equality with
  the baseline output hash

FlashInfer's deterministic baseline overflowed its prefill workspace on this
Blackwell GPU. The qualification was therefore rerun consistently with Triton,
which SGLang documents as a deterministic backend for this model family. Values
below must not be compared with a faster, nondeterministic FlashInfer baseline.

## Results

| Variant | Median TTFT | Median decode | Median end-to-end | Decode speedup | Total speedup | Exact-output result |
|---|---:|---:|---:|---:|---:|---|
| Baseline | 85.9 ms | 60.28 tok/s | 4.316 s | 1.00× | 1.00× | Reference; stable |
| N-gram | 106.0 ms | 331.06 tok/s | 0.876 s | 5.49× | 4.93× | Failed: 2 hashes; no baseline match |
| EAGLE3 compiled | 50.4 ms | 95.92 tok/s | 2.709 s | 1.59× | 1.59× | Failed: stable, but not baseline match |
| Qwen3-0.6B standalone draft | 161.0 ms | 39.38 tok/s | 6.636 s | 0.65× | 0.65× | Failed: stable, but not baseline match |

The strict hash gate is intentionally conservative. A divergent hash does not
necessarily mean unusable text quality, but it means the run cannot support the
claim that speculative verification made the *same generation* faster. Future
quality-oriented work should compare task quality or distributional metrics as
a separate experiment, not weaken this timing gate after seeing results.

## Interpretation

N-gram speculation benefits when the prompt or generated continuation repeats
token sequences already present in the context. A prior synthetic repeated
prompt reached 4.62× decode speedup and 4.00× total speedup, showing why that
prompt is too favorable for a general claim. On the realistic project-document
prompt, n-gram was faster but unstable.

Compiled EAGLE3 reduced both decode and total latency, but the exact token stream
diverged. The standalone 0.6B draft added more draft/verification overhead than
it saved and was slower than direct decoding.

The practical conclusion is that server-side speculative settings alone are
unlikely to deliver the requested 10× for one 1,000-token-prompt request on this
model and GPU. A credible next phase should test a model/hardware change—such as
an FP8 checkpoint on the fast FlashInfer path—then independently add EAGLE-style
speculation, measuring quality and exact-output performance as separate claims.

## Cost and cleanup

The successful GPU instance ran from 02:54:21 to 03:19:23 UTC, approximately
25 minutes. At the recorded $3.9982/hour on-demand rate, its EC2 cost was about
$1.67, excluding small S3, EBS, transfer, tax, credit, and discount effects.
The runner destroyed ten run-scoped resources and verified that no live EC2
instance or EBS volume remained for run `dai-gen-20260822-025406`.

Raw machine-readable results are retained locally under
`prototype/results/aws-generation/dai-gen-20260822-025406/` and excluded from
Git.
