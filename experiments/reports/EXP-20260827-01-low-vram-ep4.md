# EXP-20260827-01 — Four-node low-VRAM expert parallelism

## Metadata

- Date: 2026-08-27
- Status: `negative` for performance; `validated` for joint capacity and execution
- Successful run ID: `dai-ep4-20260827-121202`
- Failed setup run ID: `dai-ep4-20260827-115653`
- Code revision: `d3ad86c` on `codex/low-vram-ep`
- Raw artifacts: `prototype/results/aws-low-vram-ep/dai-ep4-20260827-121202/`

## Question and hypothesis

Can GPUs with no more than 24 GiB VRAM jointly hold Qwen3-30B-A3B BF16 and improve or preserve useful single-request performance?

The capacity hypothesis passes if no worker can hold the full checkpoint, all four ranks initialize, every rank participates in one inference, and peak allocation remains within physical VRAM. The performance hypothesis passes only if the distributed cell is competitive with the same cold 1,000-input/256-output single-GPU baseline.

## Fixed contract

- Model: pinned Qwen3-30B-A3B BF16 checkpoint, approximately 57 GiB
- Runtime: `lmsysorg/sglang:v0.5.16`
- Hardware: 4 × AWS `gr6.4xlarge`, one NVIDIA L4 with 22,888 MiB VRAM per worker
- Topology: one AZ and cluster placement group; NCCL socket transport without EFA
- Parallelism: TP4 / DP4 / EP4 with DP attention
- Workload: batch 1, exactly 1,000 input tokens and 256 output tokens
- Cache: cold; radix cache flushed before every request after the initializing warmup
- Sampling: temperature zero, fixed seed; production-fast nondeterministic inference path
- Repetitions: two warmups and ten measured requests
- Comparison: qualified cold target-only baseline, 168.246 pooled tok/s

## Results

| Metric | Four-node EP4 | Single-GPU baseline | Comparison |
|---|---:|---:|---:|
| Pooled output throughput | 12.658 tok/s | 168.246 tok/s | 0.0752× |
| Mean cold TTFT | 626.08 ms | 66.06 ms | 9.48× slower |
| Mean end-to-end latency | 20.772 s | 1.582 s | 13.13× slower |
| Peak VRAM per worker | 20,869 MiB | — | 91.2% of physical VRAM |

All measured requests returned 256 tokens. The distributed run produced four unique output hashes and did not match the single-GPU token hash.

Every rank's network counters increased after startup. Rank deltas were 19,585.6, 19,349.6, 19,134.5, and 19,337.3 MiB, totaling 81,167,054,405 bytes across the smoke and full evaluations.

The first attempt proved that BF16 weights fit—15.96 GiB loaded with 5.73 GiB initially free—but failed because `max-running-requests=1` was divided across DP4 and rounded to zero. Setting it to four produced one request slot per DP rank and allowed the server and benchmark to complete.

## Validity boundary

This proves true joint full-model inference on four 24GB-class GPUs: no node can hold the checkpoint, the server resolved TP4/DP4/EP4, all ranks exchanged traffic, and peak memory stayed under the hardware limit.

It does not establish exact token parity or agentic-coding quality. Deterministic inference was disabled, and four temperature-zero hashes appeared across ten runs. The descriptive p99 fields in the raw report have only ten observations and are not production SLA estimates.

The raw benchmark `variant` says `ep8`; this is a label-only bug inherited from the earlier 12 GiB design. The run manifest, four AWS instance records, four rank logs, and resolved server configuration prove the EP4 topology. The runner is corrected for future runs.

## Decision

Capacity passed and performance failed. Ordinary socket-based cross-host collectives overwhelm compute for this token-by-token EP4/TP4 topology. Do not scale this exact design by adding more identical Ethernet-connected workers.

Continue only with experiments that reduce communication per token or materially improve the transport: hot-expert replication/caching, expert quantization that reduces worker count, or EFA/GPUDirect-class collectives.

## Follow-ups

- [ ] Measure routed expert frequency and bytes per layer/token, then preregister a hot-expert replication policy.
- [ ] Test the smallest quality-qualified quantization that fits the model on two 24GB-class workers.
- [ ] Run the same topology on an EFA-capable instance family to isolate transport from parallelization overhead.
- [ ] Apply paired agentic-coding quality evaluation before making a production-quality claim.

## Cost and cleanup

- Launch-to-termination-request compute estimate: approximately $2.30 across the failed discovery attempt and successful run, excluding small EBS and termination-transition charges
- Planned four-node rate: $6.157/hour
- Automatic expiry: independent three-hour timer on every node
- Teardown: both OpenTofu stacks destroyed
- Independent audit: zero live EP4 instances, zero EP4 EBS volumes, and empty OpenTofu state

## Evidence

- [Human-readable raw summary](../../prototype/results/aws-low-vram-ep/dai-ep4-20260827-121202/README.md)
- [Machine-readable report](../../prototype/results/aws-low-vram-ep/dai-ep4-20260827-121202/report.json)
- [Qualified target-only baseline](../../prototype/TARGET-ONLY-100-TPS-2026-08-24.md)
