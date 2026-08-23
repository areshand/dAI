# Single-request heterogeneous scaling: execution report

Date: 2026-08-22

## Decision

The current system does **not** have a correctness-qualified path to 10x
single-request acceleration for Qwen3-30B-A3B. The fastest raw result was
NGRAM-16 at 369.20 tok/s (6.39x decode throughput), but every speculative
variant produced a different full-output token hash from the single-GPU
reference. The only exact-output result was the 59.04 tok/s baseline.

Adding machines was therefore not tested at one/two/four-worker scale. This is
an intentional gate, not missing data: no one-worker direction first passed the
local correctness and minimum-improvement prerequisites.

The most promising remaining mechanism is remote asynchronous drafting over a
sub-millisecond link, but it is not yet a measured win. With the measured Wi-Fi
RTT it models to only 1.017x baseline, below the 1.10x research gate. The current
SGLang remote-drafter path is also still upstream roadmap work rather than a
production feature.

## Benchmark contract

- Target: Qwen3-30B-A3B, batch one, 1,000-token prompt, 256 generated tokens.
- Reference runtime: SGLang v0.5.16 on one `g7e.4xlarge`, NVIDIA RTX PRO 6000
  Blackwell 96 GB, `us-east-2b`.
- Confirmatory paired baseline: 59.036 tok/s, 16.939 ms mean token period,
  17.069/17.121 ms client ITL p95/p99, 70.326 ms mean TTFT.
- Correctness gate: one stable full-output token hash equal to baseline.
- Latency gate: client ITL p95 and p99 below 100 ms.
- Minimum worthwhile gate: at least 10% lower time/token than the paired
  baseline.
- Scaling gate: zero/one/two/four **added** machines, only after one added
  machine passes correctness and minimum improvement.

## Full-model results

| Variant | Mean tok/s | Decode speedup | Stable within variant | Exact baseline hash | Gate result |
|---|---:|---:|:---:|:---:|---|
| Baseline | 59.04 | 1.00x | yes | yes | reference |
| NGRAM | 304.63 | 5.51x | no (2 hashes) | no | fail |
| NGRAM-16 | 369.20 | 6.39x | yes | no | fail |
| NGRAM-32 | 272.72 | 5.47x | no (6 hashes) | no | fail |
| EAGLE3, uncompiled | 60.77 | 1.03x | yes | no | fail |
| EAGLE3, compiled | 94.78 | 1.60x | yes | no | fail |
| Standalone Qwen3-0.6B draft | 38.62 | 0.65x | yes | no | fail |

NGRAM streaming coalesced multiple output tokens into SSE events, so exact
per-token ITL cannot be reconstructed. The harness retains raw event timestamps
and reports event interarrival instead of inventing sub-event timing. The
speculative variants' p99 event gaps were 23–64 ms; this is useful transport
evidence but is not relabeled as token ITL.

## Direction 1: asynchronous remote draft/verify

### Draft compute and transport

| Draft cell | One-token decode | Four-token round | Decision |
|---|---:|---:|---|
| Coordinator CPU, eager | 24.56 ms | 96.49 ms | no-go |
| Intel Mac `x`, eager CPU | 186.08 ms | 745.68 ms | no-go |
| AWS L4, eager synchronized | 27.11 ms | 108.50 ms | launch-bound proxy; no-go |
| AWS L4, SGLang decode graph | 6.010 ms | 24.04 ms | viable component |

The optimized L4 control achieved 166.37 tok/s, 6.125/6.190 ms ITL p95/p99,
and 25.424 ms TTFT p99. This shows why the earlier eager-HuggingFace GPU result
was misleading: synchronizing every token measured launch overhead, while
SGLang's decode graph is 4.5x faster.

Measured 64 KiB Wi-Fi remote-minus-local p50 was 5.704 ms. The standalone
request-aligned trace reported:

- 155 correct draft tokens from 404 proposed tokens;
- 101 target verification cycles for 256 output tokens;
- 2.535 output tokens per verification cycle;
- acceptance histogram `[37, 20, 17, 7, 20]`, hence 20/101 full four-token
  windows;
- 62.54 ms observed cycle in the native repeated run.

The acceptance-aware screen uses the observed full-window fraction and accepted
length, not the aggregate acceptance rate as an independent per-token alpha:

`cycle = p_full * max(draft, RTT + verify) + (1 - p_full) * (draft + RTT + verify)`

Using the optimized 24.04 ms L4 draft round, measured 5.704 ms Wi-Fi RTT, and an
optimistic 16.939 ms one-baseline-step verification cost gives 42.20 ms/cycle,
or 16.65 ms per accepted token. That is only 1.017x baseline and misses the
15.245 ms/token 10%-improvement boundary. At assumed zero RTT the screen is
14.84 ms/token (1.14x), which makes sub-millisecond transport a narrow
hypothesis, not a measured result.

No async remote prototype or two/four-drafter experiment was run because no
measured component cell passed the gate. This also avoids mistaking SGLang's
present decoupled-spec flags for a complete system: the upstream
[parallel speculative decoding roadmap](https://github.com/sgl-project/sglang/issues/27462)
still lists verifier execution, drafter enumeration, synchronized E2E, and
overlap as unfinished, while the earlier
[SPECTRE proposal](https://github.com/sgl-project/sglang/issues/22044) is an RFC.

## Direction 2: multi-drafter/token-tree verification

Native request-aligned instrumentation produced:

| Variant | Correct/proposed | Verify cycles | Output/cycle | Full-window count |
|---|---:|---:|---:|---:|
| EAGLE3 uncompiled | 121/402 | 134 | 1.910 | 17/134 |
| EAGLE3 compiled | 115/426 | 142 | 1.803 | 15/142 |
| Standalone 0.6B | 155/404 | 101 | 2.535 | 20/101 |

Uncompiled EAGLE3 first differs from baseline at output token 14; compiled
EAGLE3 first differs at token 92. Compilation therefore changes the numerical
path but is not the root cause. NGRAM, NGRAM-16, NGRAM-32, standalone draft, and
uncompiled EAGLE3 all first diverge at token 14, implicating a shared
speculative/batch-shape path.

A follow-up target-logprob run isolated that path. Ordinary generation routes
the target through Triton decode attention, while `TARGET_VERIFY` routes it
through Triton extend attention (the unified extend kernel in deterministic
mode). Their BF16 logits differ starting with the first post-prefill step. At
position 14, an exact reported baseline tie between token IDs 20317 and 34920
became a 0.125 log-prob lead for 34920 in target-verify mode, flipping greedy
`argmax`. One-token drafts, disabling CUDA graphs, and both speculative
attention-mode flags produced the same alternative stream. See
[`SPECULATIVE-DIVERGENCE-ROOT-CAUSE-2026-08-23.md`](SPECULATIVE-DIVERGENCE-ROOT-CAUSE-2026-08-23.md).

Tree-node count, expert union, model bytes, and remote drafter redundancy were
not measured after the exact-output prerequisite failed. Scaling this path
would only scale a computation that cannot yet reproduce the reference.

## Direction 3: sharding/offload-and-cache

This direction is not applicable to the current comparison. The complete model
is resident on the 96 GB target GPU. Artificially lowering VRAM or switching to
a larger model would answer a capacity question, not whether remote experts
beat the best resident baseline.

CPU/NVMe offload, remote expert dispatch, hot-expert replication, latency-aware
placement, and one/two/four-worker scaling should be reopened only after
selecting a model that genuinely cannot stay resident. The correct comparator
then becomes the best zero-network CPU/NVMe offload implementation.

## Instrumentation delivered

- Generation schema v2 with raw SSE arrival timestamps, event deltas, exact
  tokenization checks, valid-only client token ITL, p95/p99, and coalescing
  counters.
- Native SGLang speculative probe with accepted/proposed drafts, verification
  count, accept length/rate, histogram, and output hash per request.
- Bootstrap confidence intervals, exact-hash gate, ITL deadline misses,
  equivalence/minimum-improvement gates, and machine-count metadata.
- Eager draft profiler, optimized SGLang L4 control, and acceptance-aware async
  screening model.
- AWS run isolation, hourly cost ceiling, hard TTL, S3 result capture, and
  teardown leak checks.

## Operational issues resolved

- `x.local` discovery was transient; direct `192.168.1.153` with the existing
  host-key alias was used. The temporary transport service was stopped.
- `g7e.4xlarge` capacity was unavailable in `us-west-2`; CloudTrail confirmed
  `Server.InsufficientInstanceCapacity`, and the full run moved to
  `us-east-2b` while reading the protected model cache.
- L4 deterministic persistent Triton matmul and default prefill graph exceeded
  shared-memory limits. The speed-only control used seeded ordinary kernels,
  disabled prefill graphs, and retained decode graphs. It is not used for output
  equivalence.
- All capacity retries, failed runs, and successful runs completed teardown.
  Final state: zero live dAI EC2 instances and zero run-scoped EBS volumes.

## Recommended next research milestone

Do not add more machines yet. First obtain both:

1. an exact-output speculative verification path with a token-by-token logit
   lock against baseline; and
2. a real remote-drafter runtime on a measured sub-millisecond link.

Then repeat the one-drafter gate. Only if it reaches at least 1.10x with exact
output should the experiment expand to two and four deadline-cancelled lanes.
For expert placement research, choose a non-resident target model first and
compare against local offload.

## Local raw artifacts

The following ignored paths preserve the raw evidence on the experiment host;
the shareable aggregate results and decisions are recorded in this report.

- Full-model run: `prototype/results/aws-generation/dai-gen-20260822-185155/`
- Optimized L4 draft run: `prototype/results/aws-generation/dai-gen-20260822-201516/`
- Research gates: `prototype/results/aws-generation/dai-gen-20260822-185155/research-gates.json`
- Comparison: `prototype/results/aws-generation/dai-gen-20260822-185155/comparison.json`
- Closed ledger: `prototype/RESEARCH-TODO-2026-08-22.md`
