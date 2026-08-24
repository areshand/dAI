# Target-only 100 tok/s qualification

Date: 2026-08-24

## Decision

The Qwen3-30B-A3B testbed exceeds 100 output tokens/second without speculative
decoding, quantization, a remote drafter, or multiple machines. A fresh
cold-cache confirmation measured **168.246 pooled output tok/s** on one
`g7e.4xlarge` (NVIDIA RTX PRO 6000 Blackwell 96 GB).

The earlier 58–60 tok/s reference was a numerical-debugging configuration. It
enabled SGLang's batch-invariant deterministic inference and pinned Triton
attention, which also selected PyTorch sampling and deterministic MoE kernels.
Two older production-fast runs had already measured about 170 tok/s. The new
run confirms that result under the corrected cold-prompt contract.

## Confirmatory contract

- Model: pinned Qwen3-30B-A3B BF16 checkpoint
- Runtime: pinned `lmsysorg/sglang:v0.5.16`
- Hardware: one `g7e.4xlarge` in `us-east-2b`
- Parallelism and load: TP1, batch one, no competing requests
- Input/output: exactly 1,000 input tokens and 256 output tokens
- Sampling: temperature zero, ignore EOS, fixed server seed 1234
- Repetitions: two warmups and ten measured requests
- Cache: radix cache retained, but successfully flushed before every request
  after the first warmup; flush latency is excluded from TTFT
- Resolved server path: FlashInfer attention and sampling, CUDA graphs enabled,
  deterministic inference disabled, no speculative algorithm, no quantization

## Result

| Metric | Result |
|---|---:|
| Pooled steady-decode throughput | **168.246 tok/s** |
| Per-request mean / p50 throughput | 168.246 / 168.242 tok/s |
| Per-request min / max throughput | 168.175 / 168.327 tok/s |
| One-sided 95% t lower bound on mean throughput | **168.217 tok/s** |
| Mean / p95 TTFT | 66.06 / 66.34 ms |
| Mean / p95 end-to-end latency | 1.5817 / 1.5826 s |
| Exact output length | 256 tokens in 10/10 measured requests |
| Within-run output stability | one output hash in 10/10 requests |

The server log contains twelve requests with `#new-token: 1000` and
`#cached-token: 0`, covering both warmups and all ten measured repetitions. The
report records eleven successful cache flushes: every request after the first
initializing warmup.

The lower confidence bound is 68.2% above the 100 tok/s target. The result is
also the third independent production-fast run above 100 tok/s: the two prior
warm-cache runs measured 169.657 and 169.916 mean tok/s. Their decode results
were valid, but their roughly 54 ms TTFT values must not be used as cold
1,000-token prefill evidence.

## Quality boundary

This cell is target-only BF16 inference, so it has no lossy draft acceptance,
quantization, or surrogate model. It therefore establishes production target
speed, not speculative quality equivalence. FlashInfer and deterministic
Triton can choose different greedy tokens at numerical near-ties, so one stable
prompt hash is not a comprehensive agentic-coding quality evaluation. The full
agentic suite remains a separate qualification item.

Remote drafting remains useful research only if it beats this optimized
168 tok/s target baseline or improves fleet economics/throughput. Comparing a
remote drafter to the 59 tok/s deterministic diagnostic would overstate its
benefit.

## Evidence and cleanup

Raw local evidence is retained in the ignored directory
`prototype/results/aws-generation/dai-gen-20260824-065952/`:

- `baseline.json`: schema-v2 traces and aggregate metrics
- `baseline-server-info.json`: resolved runtime configuration
- `baseline-server.log`: cold-token accounting and server telemetry
- `comparison.json`: single-cell comparison artifact

OpenTofu destroyed all ten run-scoped AWS resources. The runner and independent
post-run queries both found zero remaining instances and zero EBS volumes for
run `dai-gen-20260824-065952`.
