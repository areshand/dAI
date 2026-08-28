# EXP-20260828-02 — Sparse EP output combine

## Metadata

- Date: 2026-08-28
- Status: `negative` for performance; `validated` for sparse-combine execution
- Run ID: `dai-ep4-sparse-combine-20260828-113000`
- Owner: Bo Wu / Codex
- Code revision: `codex/expert-placement-locality`, based on `95efcbf`
- Raw artifacts: `prototype/results/aws-low-vram-ep/dai-ep4-sparse-combine-20260828-113000/`
- Upstream context: [SGLang issue #36820](https://github.com/sgl-project/sglang/issues/36820)

## Question and hypothesis

Can SGLang avoid waiting for expert-parallel ranks that own none of a decoded token's selected experts, and does that make trace-optimized expert co-location faster on four 24 GB GPUs?

The mechanism hypothesis passes if single-token layers with fewer than four active expert ranks use only those contributors plus the token owner, dense or incompatible cases fall back to stock `reduce_scatterv`, and all measured requests complete with 256 non-collapsed server token IDs. The performance hypothesis passes if optimized placement exceeds bracketed sparse/trivial TPS and measured network traffic falls consistently, with both stock and sparse control drift below 5%.

## Plans considered

1. **Dynamic active-rank subgroups.** Pre-create collective groups for possible active-rank sets and reduce only within the selected subgroup. This is tractable for EP4 but scales poorly, requires the token owner to join every chosen group, complicates CUDA graphs and multi-owner batches, and still leaves the full input gather.
2. **Sparse point-to-point output combine.** Keep SGLang's existing router, expert kernels, and input gather; during batch-one decode, send partial MoE output only from active expert ranks to the token owner. This was selected as the smallest end-to-end prototype that directly tests the missing placement mechanism.
3. **GPU-resident source-side sparse activation dispatch.** Route activations from the token owner only to selected expert ranks, execute local experts, and return/aggregate contributions. This is the intended complete design, but it requires a real dispatcher, GPU metadata/control, multi-owner packing, and CUDA-graph-aware fallbacks.
4. **Existing DeepEP/NCCL EP backends.** These remain important production comparisons, but the pinned SGLang v0.5.16 L4/ENA socket topology does not provide the RDMA/NVLink or newer NCCL/nccl4py environment needed for a like-for-like first test.
5. **Collective tuning alone.** A faster full-rank reduce-scatter can improve the baseline but cannot turn better expert co-location into fewer participants, so it does not answer the placement question.

Plan 2 was selected because it changed the exact output-combine operation implicated by the previous experiment while retaining a safe stock fallback. The run falsified it as a performance design and strengthens the case for Plan 3.

## Implementation

The pinned SGLang v0.5.16 Qwen3 normal-forward path records the physical ranks containing the eight selected experts only when the gathered global MoE buffer contains one decode token. The downstream layer communicator then:

- infers the single token-owning DP rank from `reduce_scatterv` sizes;
- falls back to stock `reduce_scatterv` for prefill, multi-token/multi-owner shapes, invalid topology, or all-four-rank fanout;
- creates a dedicated NCCL communicator once, avoiding operation-order conflicts with the TP communicator;
- has only active expert ranks send their partial hidden state to the token owner;
- lets non-contributing ranks skip the output combine and advance to the next layer boundary;
- records eligible calls, sparse calls, dense fallbacks, active-rank histograms, and direct payload bytes.

The eight physical expert IDs are copied from GPU to CPU once per MoE layer to choose the branch. That synchronization is intentionally included in prototype latency and is not proposed as a production implementation.

## Fixed contract

- Model/checkpoint: pinned Qwen3-30B-A3B BF16, 48 MoE layers, 128 experts/layer, top 8/token
- Runtime/image: SGLang v0.5.16; source-hash-pinned Qwen3 placement and sparse-communicator patches
- Backends: FlashInfer attention/sampling, `moe_a2a_backend=none`, CUDA graphs disabled
- Hardware: 4 × AWS `gr6.4xlarge`, one NVIDIA L4 with 22,888 MiB VRAM per worker
- Topology: one AZ, cluster placement group, NCCL socket transport over ENA, no EFA
- Parallelism/load: TP4 / DP4 / EP4 with DP attention; batch one; no concurrent load
- Prompt: fixed benchmark text, exactly 1,000 input and 256 output tokens
- Cache: cold, explicit radix-cache flush before every post-initialization request
- Sampling: temperature zero, fixed seed, production-fast nondeterministic GPU path
- Placement: trivial contiguous versus the pinned trace-optimized 48 × 128 permutation; EPLB and replication disabled
- Order: full/trivial pre, sparse/trivial pre, sparse/optimized, sparse/trivial post, full/trivial post
- Repetitions: two warmups and ten measured requests per cell; an additional 32-token sparse smoke before measurement
- Structural gate: every measured request must contain 256 server token IDs and more than one unique token ID

## Results

| Cell | Pooled TPS | Mean cold TTFT | Mean total | p95 ITL | Network bytes |
|---|---:|---:|---:|---:|---:|
| Full/trivial pre | 12.799 | 615.43 ms | 20.538 s | 82.21 ms | 81.216 GB† |
| Sparse/trivial pre | 11.479 | 623.20 ms | 22.838 s | 89.54 ms | 71.858 GB |
| Sparse/optimized | 10.662 | 633.45 ms | 24.551 s | 98.53 ms | 69.630 GB |
| Sparse/trivial post | 11.521 | 628.74 ms | 22.763 s | 89.41 ms | 71.824 GB |
| Full/trivial post | 12.581 | 618.77 ms | 20.888 s | 83.92 ms | 73.079 GB |

Bracketed stock/full TPS was **12.690**. Bracketed sparse/trivial TPS was **11.500**, a **9.38% regression**. Within sparse mode, optimized placement reached **10.662 TPS**, another **7.29% regression**, with **7.68% higher total latency** than bracketed sparse/trivial.

Control stability was acceptable: full TPS drifted -1.71% and sparse TPS drifted +0.37%, both within 5%.

### Did the sparse mechanism actually run?

Yes. The per-rank cumulative instrumentation snapshots were identical:

| Placement | Mean active ranks | Sparse calls | Dense fallbacks | Sparse fraction | Direct-send payload snapshot |
|---|---:|---:|---:|---:|---:|
| Trivial pre | 3.675 | 45,694 | 101,762 | 30.99% | 395.2 MiB |
| Optimized | 2.939 | 107,175 | 40,281 | 72.68% | 797.7 MiB |
| Trivial post | 3.675 | 45,143 | 100,777 | 30.94% | 390.3 MiB |

Optimized placement therefore changed the intended control path: full-rank combines fell from about 69.0% to 27.3%. It still got slower. The negative result is not explained by the sparse branch failing to activate.

### Network interpretation

Bracketed sparse/trivial interface traffic was 71.84 GB. Optimized sparse traffic fell 3.08% to 69.63 GB. The same-scope full/trivial post cell transferred 73.08 GB, so sparse/trivial was only 1.69% lower descriptively. The full/trivial pre network interval included the initial 32-token stock smoke and is marked †; it is not combined with full/post for a protocol traffic claim. These reductions are small relative to the route-fanout change because the full-rank activation gather, NCCL control traffic, HTTP/SSM/background traffic, and dense fallbacks remain.

The optimized map caused more layers to take the P2P branch, so cumulative direct-send payload increased even though active-rank fanout fell. On this small-message socket topology, many direct sends plus one GPU-to-host synchronization per layer were slower than SGLang/NCCL's optimized four-rank reduce-scatter.

## Correctness and validity boundary

The distributed smoke and all fifty measured requests returned the requested 256 server token IDs without repeated-token collapse. This validates execution and the performance comparison.

It does **not** establish lossless output equivalence or agentic-coding quality. Each cell produced four output hashes, and the full/trivial and sparse/trivial hash sets did not overlap. Sparse accumulation uses a different BF16 reduction order, so temperature-zero argmax can change at near-ties; a behavioral difference is not automatically a quality regression, but it is also not eligible to be called lossless without logit/tensor tolerance and paired task evaluation.

The study uses one prompt, one EP4 topology, no concurrent load, and L4/ENA socket networking. It does not characterize EFA/GPUDirect, DeepEP, larger batches, request cohorts, other models, or geographically decentralized nodes. Descriptive p95 ITL comes from 2,550 token intervals per cell, not independent SLA trials.

## Decision

Stop the Python/D2H sparse output-combine plan. It technically lets inactive ranks skip that combine, but the communication unit is too fine and too late in the layer: all ranks already received the activation, and direct 4 KiB-class messages cost more than the stock collective they replace.

The most promising SGLang implementation is now a **GPU-resident source-side sparse dispatcher**, not another combine patch. The token owner should pack activations and routing metadata on GPU, send once per selected destination, run only destination experts, and return aggregated contributions. Dense/multi-owner batches should retain a tuned collective fallback. The next prototype should start with a one-layer numerical oracle and microbenchmark before another paid full-model run.

For single-request decentralized inference, expert co-location must reduce the request's active cohort closer to one or two workers; mean fanout 2.94 still creates a network dependency in nearly every layer. Hot-expert replication or cohort-aware placement should be evaluated only after source-side sparse dispatch removes the unconditional gather.

## Follow-ups

- [ ] Implement a one-layer GPU-resident dispatch microbenchmark with stock-allgather/reduce-scatter and sparse-dispatch controls.
- [ ] Keep active-rank discovery on GPU; measure control synchronization, message count, bytes, kernel time, and end-to-end layer time separately.
- [ ] Add multi-owner packing and a preregistered density threshold that selects sparse or collective transport without changing semantics.
- [ ] Add an unsharded expert-output numerical oracle, deterministic top-logprob probe, and paired agentic-coding quality gate.
- [ ] Explore hot-expert replication/cohort placement only after the dispatcher passes; target one-to-two-rank mean fanout.
- [ ] Compare an EFA/GPUDirect-capable backend before generalizing the L4/ENA result to expert parallelism broadly.

## Cost and cleanup

- Planned compute rate: $6.157/hour for four nodes
- Approximate compute cost: $4.5 for roughly 44 minutes, excluding small EBS/network charges
- Automatic expiry: independent three-hour shutdown timer on every node
- Teardown: OpenTofu destroyed all 14 run-scoped resources
- Independent audit: zero live tagged instances and zero tagged EBS volumes

## Evidence

- [Machine-readable comparison](../../prototype/results/aws-low-vram-ep/dai-ep4-sparse-combine-20260828-113000/sparse-ep-combine-comparison.json)
- [Run manifest](../../prototype/results/aws-low-vram-ep/dai-ep4-sparse-combine-20260828-113000/manifest.json)
- [Full/trivial pre](../../prototype/results/aws-low-vram-ep/dai-ep4-sparse-combine-20260828-113000/benchmark.json)
- [Sparse/trivial pre](../../prototype/results/aws-low-vram-ep/dai-ep4-sparse-combine-20260828-113000/sparse-trivial-pre-benchmark.json)
- [Sparse/optimized](../../prototype/results/aws-low-vram-ep/dai-ep4-sparse-combine-20260828-113000/sparse-optimized-benchmark.json)
- [Sparse/trivial post](../../prototype/results/aws-low-vram-ep/dai-ep4-sparse-combine-20260828-113000/sparse-trivial-post-benchmark.json)
- [Full/trivial post](../../prototype/results/aws-low-vram-ep/dai-ep4-sparse-combine-20260828-113000/full-trivial-post-benchmark.json)
- Runtime source patch: `scripts/apply_sglang_sparse_ep_combine_patch.py`
- Harness: `scripts/aws-low-vram-ep-experiment.sh`

## Corrections

None.
