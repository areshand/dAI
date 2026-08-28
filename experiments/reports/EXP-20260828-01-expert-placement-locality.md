# EXP-20260828-01 — Expert locality under full collectives

## Metadata

- Date: 2026-08-28
- Status: `negative` for performance; `validated` for held-out placement locality
- Corrected run ID: `dai-ep4-placement-fixed-20260828-054500`
- Invalid precursor run ID: `dai-ep4-placement-20260828-044700`
- Owner: Bo Wu / Codex
- Code revision: `codex/expert-placement-locality` PR branch
- Raw artifacts: `prototype/results/aws-low-vram-ep/dai-ep4-placement-fixed-20260828-054500/`

## Question and hypothesis

Can routed-expert traces be used to co-locate frequently co-activated experts on fewer of four 24 GB workers, reducing cross-host work and improving batch-one Qwen3-30B-A3B inference?

The placement hypothesis passes if the optimized map lowers held-out worker fanout and activation imbalance. The performance hypothesis passes only if optimized pooled output TPS exceeds the average of bracketing trivial-placement runs while the baseline drift remains within 5%, every measured request returns exactly 256 non-degenerate token IDs, and the runtime resolves the intended static map without replication or EPLB.

## Fixed contract

- Model/checkpoint: pinned Qwen3-30B-A3B BF16, 48 MoE layers, 128 experts per layer, top 8 experts per token
- Runtime: SGLang v0.5.16 commit `fdebc938`, image digest `sha256:7b6a35df9839fd593a94a1eaee82d7777f472225d9f3ad1f8a2e0cb2bd1785d0`
- Runtime correction: pinned `qwen3_moe.py` source SHA-256 `b18eb188c...bc2`, patched SHA-256 `43bc70df...84e0`
- Resolved backends: FlashInfer attention and sampling, `moe_a2a_backend=none`, CUDA graphs disabled
- Hardware: 4 × AWS `gr6.4xlarge`, one NVIDIA L4 with 22,888 MiB VRAM per worker
- Topology: one AZ and cluster placement group; NCCL socket transport without EFA
- Parallelism/load: TP4 / DP4 / EP4 with DP attention, batch one, one request slot per DP rank
- Prompt: fixed benchmark text, exactly 1,000 input tokens and 256 output tokens
- Cache: cold; radix cache flushed before each post-initialization request
- Sampling: temperature zero, fixed seed, production-fast nondeterministic inference path
- Placement capacity: exactly 32 experts per layer per worker; one copy of every expert; EPLB disabled
- Optimization split: first five measured routing requests for fitting, last five held out
- Execution order: trivial pre, optimized, trivial post
- Repetitions: two warmups and ten measured requests per full cell
- Structural correctness gate: all measured outputs contain exactly 256 token IDs and more than one unique token ID

## Placement method

For every layer, the optimizer begins with an activation-balanced assignment. It then performs capacity-preserving expert-pair swaps that reduce the cut weight of frequently co-activated expert pairs while enforcing a 1.05× training-load ceiling. The resulting SGLang `physical_to_logical_map` is a 48 × 128 matrix; every layer is a permutation of logical expert IDs and every worker retains exactly 32 physical slots.

The custom mapping exposed an SGLang v0.5.16 Qwen3 issue: `forward_normal()` loaded weights into remapped physical slots but did not pass expert-location dispatch metadata to `topk()`. The runner now extracts the exact pinned source from the image, refuses an unexpected source hash, adds logical-to-physical dispatch metadata, mounts the corrected file read-only, and records its patched hash on every rank.

## Results

### Held-out routing prediction

| Metric | Trivial | Optimized | Change |
|---|---:|---:|---:|
| Mean workers per token-layer | 3.655 | 2.952 | -19.2% |
| Token-layers involving all four workers | 67.37% | 26.23% | -41.14 pp |
| Cross-worker co-activated expert pairs | 76.00% | 59.12% | -16.88 pp |
| Mean layer max/mean activation ratio | 1.298× | 1.049× | better balance |
| Worst layer max/mean activation ratio | 1.776× | 1.059× | better balance |

### Measured performance

| Cell | Pooled output TPS | Mean cold TTFT | Mean end-to-end latency |
|---|---:|---:|---:|
| Trivial pre | 12.507 | 642.19 ms | 21.030 s |
| Optimized | 11.909 | 639.10 ms | 22.052 s |
| Trivial post | 12.584 | 632.33 ms | 20.897 s |
| Bracketed trivial | 12.545 | 637.26 ms | 20.964 s |

- Optimized speed ratio versus bracketed trivial: **0.9492×**, a **5.08% regression**.
- Optimized TTFT ratio versus bracketed trivial: 1.0029×, effectively unchanged.
- Trivial pre/post TPS drift: 0.61%, within the preregistered 5% stability bound.
- All measured requests in all three cells returned 256 token IDs and non-degenerate sequences.
- Each cell produced four output hashes. The optimized hash set differed from both trivial cells; the two trivial hash sets matched one another.

### Network evidence

The equally scoped trivial-pre and optimized phases each include the same smoke and full benchmark requests. Summed non-loopback interface deltas were:

- Trivial pre: 81,146,232,210 bytes
- Optimized: 81,249,169,055 bytes
- Difference: +102,936,845 bytes, or +0.13%

The optimized map therefore did not reduce physical network traffic. The smaller 73.24 GB trivial-post counter is not directly comparable because that phase intentionally omitted the extra smoke cell.

## Validity boundary

This run validates that the fitted map substantially reduces predicted worker fanout and compute imbalance on five held-out requests. It also provides a valid structural performance comparison: the corrected remapping path returned complete non-degenerate outputs, the runtime resolved the requested configurations, and the bracketing control remained stable.

It does not prove exact token parity, lossless answer equivalence, or agentic-coding quality. Deterministic inference was disabled, and the temperature-zero GPU path produced four hashes per cell. It also uses one fixed prompt; expert co-activation stability across repositories, prompts, and context lengths remains unknown.

Most importantly, this does not show that expert co-location is intrinsically ineffective. It shows that placement alone cannot exploit locality under the tested `moe_a2a_backend=none` path: all four ranks still participate in full collectives and the post-expert reduction. Logical fanout fell, but collective participants and network bytes did not.

## Decision

The placement-locality hypothesis passed and the performance hypothesis failed. Do not spend another paid run comparing placement heuristics on this full-collective backend. The next gate must change the communication mechanism so a worker with no selected expert sends zero activation payload and does not block the token-layer critical path.

Placement remains promising only as part of a combined design: sparse destination-aware dispatch first, then capacity-aware co-location, hot-expert replication, and topology-aware scheduling.

## Follow-ups

- [ ] Build a one-layer or small-model sparse dispatch prototype in which uninvolved workers transmit zero activation bytes.
- [ ] Require network bytes to fall when held-out worker fanout falls, while matching the unsharded expert output within a preregistered numerical tolerance.
- [ ] After sparse dispatch passes, test hot-expert replication and one- or two-worker request cohorts.
- [ ] Train and evaluate placement on diverse agentic-coding prompts and repositories rather than one fixed prompt.
- [ ] Add deterministic token/logit probes and paired task-level quality noninferiority before declaring a performance winner quality eligible.
- [ ] Isolate the 5.08% regression into dispatch-remapping overhead versus physical expert memory-access effects.

## Cost and cleanup

- Planned four-node compute rate: $6.157/hour
- Corrected run compute estimate: approximately $3.5, excluding small EBS and network charges
- Invalid precursor plus corrected-run investigation: approximately $6–7 total compute
- Automatic expiry: independent three-hour timer on every node
- Teardown: both run-scoped OpenTofu stacks destroyed
- Independent audit: zero live tagged instances and zero tagged EBS volumes after the corrected run

## Evidence

- [Comparison report](../../prototype/results/aws-low-vram-ep/dai-ep4-placement-fixed-20260828-054500/expert-placement-comparison.json)
- [Placement optimization report](../../prototype/results/aws-low-vram-ep/dai-ep4-placement-fixed-20260828-054500/expert-placement-optimization.json)
- [Optimized map](../../prototype/results/aws-low-vram-ep/dai-ep4-placement-fixed-20260828-054500/optimized-expert-placement.json)
- [Trivial pre benchmark](../../prototype/results/aws-low-vram-ep/dai-ep4-placement-fixed-20260828-054500/benchmark.json)
- [Optimized benchmark](../../prototype/results/aws-low-vram-ep/dai-ep4-placement-fixed-20260828-054500/optimized-benchmark.json)
- [Trivial post benchmark](../../prototype/results/aws-low-vram-ep/dai-ep4-placement-fixed-20260828-054500/baseline-post-benchmark.json)

## Corrections

- 2026-08-28: the precursor run `dai-ep4-placement-20260828-044700` is invalid for model-performance interpretation. Its optimized cell returned 128 re-tokenized copies of token ID 90440 despite reporting 256 server-side completion tokens. Root cause: the pinned Qwen3 normal-forward path did not translate logical router IDs to remapped physical weight slots. The result was retracted, a source-hash-pinned runtime correction and hard token gates were added, and only `dai-ep4-placement-fixed-20260828-054500` supports the conclusions above.
