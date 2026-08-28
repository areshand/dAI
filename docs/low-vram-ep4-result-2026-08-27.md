# Low-VRAM EP4 experiment — 2026-08-27

## Outcome

Qwen3-30B-A3B BF16 successfully completed full-model inference jointly across four AWS `gr6.4xlarge` workers. Each worker had one NVIDIA L4 with 22,888 MiB VRAM; no worker could hold the approximately 57 GiB checkpoint alone.

The system resolved to TP4 / DP4 / EP4 with DP attention. All four ranks initialized, loaded a shard, passed the serving health gate, exchanged network traffic, and completed a cold-cache smoke request followed by the pre-registered full benchmark.

## Contract

- Model: Qwen3-30B-A3B BF16
- Runtime: SGLang v0.5.16
- Hardware: 4 x AWS `gr6.4xlarge`, one L4 per worker
- Workload: batch 1, 1,000 input tokens, 256 output tokens
- Cache: flushed before every post-first request
- Samples: 2 warmups and 10 measured repetitions
- Comparison: existing cold-cache production-fast single-GPU result using the same prompt and output length

## Results

| Metric | Four-node EP4 | Fast single-GPU baseline | Comparison |
|---|---:|---:|---:|
| Pooled output throughput | 12.66 tok/s | 168.25 tok/s | 0.075x |
| Mean cold TTFT | 626.08 ms | 66.06 ms | 9.48x slower |
| Mean 256-token end-to-end time | 20.77 s | 1.582 s | 13.13x slower |
| Peak GPU memory | 20,869 MiB on every rank | — | 91.2% of physical VRAM |

The p99 fields in the raw report are interpolations over only ten observations and are not production p99 estimates.

## Joint-participation evidence

Network growth during the smoke and full evaluations was 19,585.6 MiB, 19,349.6 MiB, 19,134.5 MiB, and 19,337.3 MiB on ranks 0–3 respectively, or 81,167,054,405 bytes summed across ranks. This excludes server startup and model initialization because the before counters were captured after the health gate.

The server-reported configuration was `tp_size=4`, `dp_size=4`, `ep_size=4`, and `enable_dp_attention=true`. Peak allocation stayed below the observed physical-memory limit on every rank.

## Output parity

The distributed and baseline prompt hashes match, and all ten measured distributed requests returned exactly 256 output tokens. Exact output-token hashes did not match the single-GPU baseline. Despite temperature zero, the production-fast distributed run produced four unique output hashes across ten samples because deterministic inference was disabled. This run therefore establishes capacity, participation, and performance—not exact token parity or agentic-coding quality equivalence.

The raw benchmark's `variant` string still contains the obsolete text `ep8`; this is a label-only bug inherited from the first 12 GiB design. The run manifest, four instance records, four rank logs, and server-resolved TP4/DP4/EP4 configuration are authoritative. The reusable runner now emits `ep4` labels.

## Interpretation

The experiment validates capacity scaling and falsifies this particular speed strategy. Ordinary socket-based Ethernet collectives dominate token-by-token decode when both tensor and expert parallelism span four hosts. Adding more identical nodes to this topology is unlikely to improve single-request latency.

The next experiment should reduce communication per token: use expert-only sharding with less tensor parallelism, quantize expert weights so fewer workers are needed, or use a GPUDirect/EFA-class runtime and topology. Each candidate should retain the cold 1,000/256 benchmark contract and add deterministic or paired task-quality qualification.

The launch-to-termination-request compute estimate is approximately $2.30 across the failed configuration-discovery attempt and the successful run, based on the recorded instance timestamps and the pre-registered $6.157/hour four-node rate. The final AWS invoice can be slightly higher because of termination transition time and EBS usage.

Raw local evidence for run `dai-ep4-20260827-121202` is stored under `prototype/results/aws-low-vram-ep/dai-ep4-20260827-121202/` and is intentionally ignored by Git because it contains large runtime logs and traces.
