# dAI experiment journal

This journal tracks what was tested, what was learned, and which architectural directions remain credible. Newest entries appear first. Detailed measurements live in the linked reports; raw traces remain under the ignored `prototype/results/` tree.

| ID | Date | Status | Question | Decision | Report |
|---|---|---|---|---|---|
| EXP-20260828-02 | 2026-08-28 | negative | Can a sparse single-token output combine let inactive EP ranks skip communication and make optimized expert co-location faster? | The mechanism executed and cut full-rank combines from 69.0% to 27.3%, but Python/D2H control plus tiny P2P sends regressed sparse TPS 9.38% versus stock; optimized placement then regressed another 7.29%. Stop combine-only P2P and build GPU-resident source-side sparse dispatch. | [Report](reports/EXP-20260828-02-sparse-ep-combine.md) |
| EXP-20260828-01 | 2026-08-28 | negative | Can trace-optimized expert co-location reduce worker fanout and speed up four-L4 inference? | Logical locality improved substantially, but full-rank collectives preserved network traffic and optimized TPS regressed 5.08%. Require sparse destination-aware dispatch before more placement tuning. | [Report](reports/EXP-20260828-01-expert-placement-locality.md) |
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
2. Placement discovery and route locality matter only when the runtime can exploit them efficiently. The sparse-combine prototype reduced mean active ranks from 3.675 to 2.939 and made 72.68% of decode-layer combines sparse, but tiny per-layer P2P operations were slower than NCCL's four-rank reduce-scatter; optimized TPS fell to 10.662.
3. Four 24GB-class L4 workers can jointly hold and execute the BF16 model, but EP4/TP4 over ordinary Ethernet reaches only 12.69 bracketed tok/s with the stock collective. A combine-only sparse protocol regressed TPS by 9.38%; its comparable sparse cells transferred about 71.84 GB, only 1.69% below the single same-scope stock-post cell.
4. Exact token hashes and task quality are separate gates. Temperature-zero runs can diverge across valid GPU kernel paths, and agentic-coding quality still requires paired task-level evaluation.

## Next experiments

- [x] Test a combine-only sparse gate. It proved inactive ranks can skip the output combine, but failed performance because it retained the full input gather and replaced optimized NCCL collectives with many tiny host-directed P2P messages.
- [ ] Build a GPU-resident, source-side sparse activation dispatcher: the token owner must route once to selected expert ranks, avoid the full input gather, aggregate results without a host synchronization per layer, and fall back safely for dense/multi-owner batches.
- [ ] Add a one-layer numerical oracle and deterministic logit probe before another full-model run; require expert-output tolerance and task-quality eligibility separately from token-hash identity.
- [ ] Only after sparse dispatch passes, replicate or cache hot experts close to requesting token cohorts and measure whether network bytes per decoded token fall materially.
- [ ] Quantize expert weights enough to test a two-worker topology while preserving the same cold 1,000-input/256-output and task-quality contract.
- [ ] Compare ordinary socket collectives with an EFA/GPUDirect-capable topology before attributing the EP4 result to expert parallelism in general.
- [ ] Add paired agentic-coding quality evaluation for any path whose output tokens differ from the qualified baseline.

## Corrections carried forward

- 2026-08-24: cold-cache validation showed the production-fast target-only baseline is 168.246 tok/s, superseding performance conclusions that treated the 59 tok/s deterministic/Triton cell as the model's ceiling.
- 2026-08-27: the raw EP4 benchmark `variant` contains the obsolete label `ep8`. The four-instance manifest, four rank logs, and resolved TP4/DP4/EP4 server configuration are authoritative; the reusable runner now emits `ep4`.
- 2026-08-28: retracted the optimized cell from `dai-ep4-placement-20260828-044700`. The Qwen3 normal-forward path loaded remapped physical weights without translating logical router IDs, producing a degenerate repeated token. The source-hash-pinned correction and hard output gates were validated in `dai-ep4-placement-fixed-20260828-054500`.
