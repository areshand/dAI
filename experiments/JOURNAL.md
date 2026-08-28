# dAI experiment journal

This journal tracks what was tested, what was learned, and which architectural directions remain credible. Newest entries appear first. Detailed measurements live in the linked reports; raw traces remain under the ignored `prototype/results/` tree.

| ID | Date | Status | Question | Decision | Report |
|---|---|---|---|---|---|
| EXP-20260827-01 | 2026-08-27 | negative | Can four GPUs with no more than 24 GiB VRAM jointly serve Qwen3-30B-A3B BF16 with competitive single-request speed? | Capacity passed; ordinary socket-based EP4/TP4 performance failed. Reduce communication rather than add identical nodes. | [Report](reports/EXP-20260827-01-low-vram-ep4.md) |
| EXP-20260824-01 | 2026-08-24 | validated | Can the target-only model exceed 100 tok/s on one production-fast GPU path? | Yes: 168.246 pooled tok/s with a cold 1,000-token prompt. This supersedes the 59 tok/s diagnostic baseline for performance comparisons. | [Source report](../prototype/TARGET-ONLY-100-TPS-2026-08-24.md) |
| EXP-20260823-01 | 2026-08-23 | diagnostic | Why do temperature-zero speculative paths differ from ordinary decoding? | Different target-verification and decode kernels produce different BF16 logits at near-ties; temperature zero fixes argmax, not cross-kernel numerical equivalence. | [Source report](../prototype/SPECULATIVE-DIVERGENCE-ROOT-CAUSE-2026-08-23.md) |
| EXP-20260822-02 | 2026-08-22 | superseded | Do built-in speculative methods deliver a correctness-qualified 10× speedup? | No under the tested deterministic/Triton contract. Raw n-gram reached 331 tok/s but failed exact-output parity; the 60 tok/s baseline was later identified as a diagnostic configuration. | [Source report](../prototype/AWS-GENERATION-EXPERIMENT-2026-08-22.md) |
| EXP-20260822-01 | 2026-08-22 | exploratory | Which heterogeneous single-request scaling directions deserve execution? | Prioritize remote draft/verify, token-tree verification, and sharding/offload, but qualify against the corrected production-fast baseline. | [Research synthesis](../prototype/SINGLE-REQUEST-HETEROGENEOUS-SCALING-RESEARCH-2026-08-22.md) |
| EXP-20260820-03 | 2026-08-20 | exploratory | Does near placement improve a real full-model forward with one naturally routed remote expert? | Near reduced the expert boundary versus far, but five blocks could not establish a whole-forward difference against ~1.08 s of local-model noise. | [Source report](../prototype/AWS-FULLMODEL-EXPERIMENT-2026-08-20.md) |
| EXP-20260820-02 | 2026-08-20 | validated | Can latency discovery place synthetic hot experts correctly across AWS nodes? | Yes: latency-aware placement matched the hot-near oracle and cut block time by 51.8% versus hot-far, with identical outputs. | [Source report](../prototype/AWS-EXPERIMENT-2026-08-20.md) |
| EXP-20260820-01 | 2026-08-20 | exploratory | Can two LAN peers discover each other, move expert work, and preserve outputs? | Infrastructure and a real single-expert boundary were validated; this did not yet prove distributed full-model generation. | [Source report](../prototype/EXPERIMENT-2026-08-20.md) |

## Current conclusions

1. The target-only Qwen3-30B-A3B BF16 path already exceeds 100 tok/s on one `g7e.4xlarge`; 59–60 tok/s is a deterministic debugging reference, not the production performance floor.
2. Placement discovery matters at expert boundaries. The unresolved problem is making the amount and frequency of cross-host communication small enough that those placement gains survive full autoregressive decoding.
3. Four 24GB-class L4 workers can jointly hold and execute the BF16 model, but EP4/TP4 over ordinary Ethernet falls to 12.66 tok/s and transfers about 81.17 GB across rank interfaces during the evaluation.
4. Exact token hashes and task quality are separate gates. Temperature-zero runs can diverge across valid GPU kernel paths, and agentic-coding quality still requires paired task-level evaluation.

## Next experiments

- [ ] Replicate or cache hot experts close to the requesting tokens and measure whether network bytes per decoded token fall materially.
- [ ] Quantize expert weights enough to test a two-worker topology while preserving the same cold 1,000-input/256-output and task-quality contract.
- [ ] Compare ordinary socket collectives with an EFA/GPUDirect-capable topology before attributing the EP4 result to expert parallelism in general.
- [ ] Add paired agentic-coding quality evaluation for any path whose output tokens differ from the qualified baseline.

## Corrections carried forward

- 2026-08-24: cold-cache validation showed the production-fast target-only baseline is 168.246 tok/s, superseding performance conclusions that treated the 59 tok/s deterministic/Triton cell as the model's ceiling.
- 2026-08-27: the raw EP4 benchmark `variant` contains the obsolete label `ep8`. The four-instance manifest, four rank logs, and resolved TP4/DP4/EP4 server configuration are authoritative; the reusable runner now emits `ep4`.
