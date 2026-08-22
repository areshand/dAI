# Does adding heterogeneous machines speed up ONE batch-1 request? — Research synthesis (2026-08-22)

Goal under test: for one autoregressive stream of Qwen3-30B-A3B, does adding
heterogeneous machines (cheap GPUs, Macs, CPUs) reduce per-token latency while
holding visible inter-token latency (ITL) under 100ms — not fleet capacity, not
throughput-at-scale. This synthesizes a 10-specialist + 3-critic + 1-adjudicator
review pass over this repo's own AWS experiments plus external literature; it
supersedes no primary data, only reconciles what 14 independent passes converged
on and flags where they didn't.

## Central finding

**No evidence in this repo shows added machines are causal for single-stream
decode throughput.** The repo's own two ground-truth experiments settle this
directly:

- `AWS-FULLMODEL-EXPERIMENT-2026-08-20.md` moved one naturally-routed expert
  (layer 0, expert 53) local/near-AZ/far-AZ, real weights, real forwards, n=5
  paired blocks. Every 95% CI on the whole-forward delta **crosses zero**:
  near−local +4.88ms [-0.66, +11.08]; far−local +2.78ms [-1.23, +6.79];
  far−near −2.10ms [-11.43, +5.58]. Statistically null, not a win.
- `AWS-GENERATION-EXPERIMENT-2026-08-22.md` used **zero added machines** and
  still produced no correctness-qualified speedup — it answers a different
  question (local numerics) and must not be counted as evidence either way on
  the machines question.

**Why, mechanistically:** Qwen3-30B-A3B (48 layers, 128 experts/layer, top-8,
no shared experts, hidden=2048) fits one GPU at 56.87 GiB BF16. At batch=1,
decode is memory-bandwidth-bound, not compute-bound (Pope et al.,
[arxiv.org/abs/2211.05102](https://arxiv.org/abs/2211.05102)). A remote machine
has no compute-speed advantage over the resident GPU it would supplement — a
network hop is a pure additive tax on the serial per-token critical path
unless it creates genuinely concurrent work that the local machine cannot do
by itself. That concurrency test is the only thing that can make "add a
machine" causal, and it is exactly the test this repo has not yet run.

## Quantified baseline (verified, not re-derived)

`AWS-GENERATION-EXPERIMENT-2026-08-22.md`: `g7e.4xlarge` (RTX PRO 6000
Blackwell, 96GB), SGLang v0.5.16, Qwen3-30B-A3B BF16, Triton backend
(FlashInfer excluded — it overflowed its prefill workspace on this GPU;
values are not comparable to a faster FlashInfer baseline), batch=1, prompt =
first exactly 1,000 tokens of the project design doc, exactly **256**
generated tokens, seed 1234, n=10 reps/variant, exact-output-hash correctness
gate.

- Baseline: **60.28 tok/s** median decode, **85.9ms** TTFT, 4.316s end-to-end.
- Token period τ = 1000/60.28 = **16.589ms/token**. Cross-check: 255τ + TTFT =
  4,230.2ms + 85.9ms = 4,316.1ms — matches the reported 4.316s exactly (two
  independent specialists, baseline-math and critic-quant, reproduced this).
- Per-layer illustrative equal-split budget: τ/48 = **0.3456ms/layer** — an
  illustrative division of the token budget, *not* a measured per-layer
  compute floor (no experiment has isolated real per-layer compute time on
  this GPU).

| Variant | Median TTFT | Median decode | Decode speedup | Exact-output result |
|---|---:|---:|---:|---|
| Baseline | 85.9ms | 60.28 tok/s | 1.00× | Reference; stable |
| N-gram | 106.0ms | 331.06 tok/s | 5.49× | **Failed**: 2 hashes across 10 runs, neither matches baseline |
| EAGLE3 compiled | 50.4ms | 95.92 tok/s | 1.59× | **Failed**: stable hash, but ≠ baseline hash |
| Qwen3-0.6B standalone draft | 161.0ms | 39.38 tok/s | 0.65× | **Failed**: slower than no speculation |

No speculative variant cleared the correctness gate. The best raw number
(n-gram, 5.49×) is prompt-favorable and unstable; the requested 10× target
is not reached even before applying the correctness bar. EAGLE3's failure
is qualitatively different from n-gram's: it is *stable but wrong* — a
reproducible divergence, not run noise, matching Thinking Machines'
documented batch-invariance mechanism
([thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference)):
matmul/RMSNorm/attention kernels are not batch-invariant, so a batched
tree-verification step computes different logits than solo decode for the
same tokens. N-gram's failure is a separate mechanism — its draft length
varies per run, so its batch shape (and numerics) varies per run too. Two
distinct root causes, not one bug — four specialists converged on this
independently.

**Measured near/far expert-boundary data** (`AWS-FULLMODEL-EXPERIMENT-2026-08-20.md`,
real Qwen3-30B-A3B expert-53 calls, `c7i.xlarge` CPU workers, n=20 discovery
calls, n=5 randomized three-path blocks, seed 20260820):

| Worker | Boundary p50 | Boundary p95 | Worker compute p50 |
|---|---:|---:|---:|
| Near (same AZ + placement group) | 0.835ms | 1.330ms | 0.390ms |
| Far (cross-AZ, same region) | 2.160ms | 2.250ms | 1.040ms |

Far was 2.6× near at the boundary level — a real, solid per-hop effect. But
the whole-forward means (local 1,078.84ms, near 1,083.72ms, far 1,081.63ms)
show every paired delta crossing zero at n=5: roughly 1.08 seconds of
whole-model execution noise dominates a single-call network effect at this
sample size. Placement is real at RPC granularity and invisible at
whole-forward granularity with today's power — not evidence against
placement, and not yet evidence for it either.

## No per-token ITL distribution exists yet — and the p95-sum fallacy

**This repo has never produced a per-token p50/p95/p99 latency distribution.**
The only percentile data anywhere is the RPC-hop discovery probe above
(n=20, p50/p95 only, **no p99**) and the whole-forward mean/median table
(n=5, no percentiles at all). `full_model_multi_worker_eval.py` runs with
`use_cache=False` (prefill-only, no decode loop); `full_model_shadow_eval.py`
keeps its remote path shadow-only, never authoritative for the actual output.
No code path in this repo produces the number a `p95/p99 < 100ms` claim
would require. Any such claim made about any candidate below is currently
**unverifiable**, not merely unproven.

One specialist (hetero-net) computed: "same-AZ p95=1.330ms × 48 layers ≈
64ms," "cross-AZ p95=2.250ms × 48 ≈ 108ms → only 44/48 layers fit under
100ms." Two independent critics (critic-quant, and the adjudicator
corroborating) flagged this as invalid percentile arithmetic — the
**p95-sum fallacy**: p95(Σxᵢ) ≠ Σp95(xᵢ). Summing 48 independent per-hop
p95 values implicitly requires all 48 hops to simultaneously hit their own
95th percentile at once — under independence that joint event has
probability ≈0.95⁴⁸ ≈ 8.6%, i.e. this arithmetic produces a >99th-percentile
worst-case envelope mislabeled "p95," not a real percentile of the sum. A
second, separate problem in the same calculation: it silently swaps the SLA
target from the measured 16.6ms/token baseline to a 100ms/token target — it
"passes" only by implicitly accepting decode collapsing from 60.28 tok/s to
roughly 10 tok/s (a 6× regression), a trade-off never stated as such. Any
number built this way should be read as a napkin upper bound on tolerable
remote-layer count, not a validated latency guarantee.

## Correctness bars, kept distinct

Three different bars are repeatedly conflated across the literature and must
not be:

1. **Exact-hash equality** — this repo's actual gate. Full-sequence output
   hash must match the reference bit-for-bit. Failed by all 3 speculative
   variants tested on 2026-08-22.
2. **Exact target distribution** — the theorem-level guarantee of rejection
   sampling (Leviathan, Kalman, Matias, ICML'23,
   [arxiv.org/abs/2211.17192](https://arxiv.org/abs/2211.17192); Chen et al.,
   DeepMind, [arxiv.org/abs/2302.01318](https://arxiv.org/abs/2302.01318), who
   caveat "preserves target distribution *within hardware numerics*").
   Proven on paper, but **not guaranteed bit-exact** on real non-batch-invariant
   kernels — exactly what this repo's EAGLE3 run demonstrates.
3. **Approximate/distributional quality** — Medusa's "typical acceptance"
   ([arxiv.org/abs/2401.10774](https://arxiv.org/abs/2401.10774)), CALM's
   conformal bound ([arxiv.org/abs/2207.07061](https://arxiv.org/abs/2207.07061)).
   Never validated here, and not licensed to substitute for bar (1) without a
   separate quality experiment — which the source report itself declines to
   run, warning explicitly against weakening the timing gate after results.

## Topic-by-topic literature map, with maturity labels

**Async/disaggregated speculative decoding, token-tree verification.** The
mechanism that lets a *second machine* do useful concurrent work on one
stream is pipelining: naive synchronous remote drafting adds full RTT to the
critical path (`t_draft + RTT + t_verify`) and is a net loss unless the
target is already slow (llama.cpp's own threshold: target <50 tok/s keeps
network overhead <1%). Genuinely asynchronous pipelining changes this to
`max(t_draft, RTT + t_verify)` — the one mechanism found anywhere in the
review where a second machine performs concurrent useful work on the *same*
stream. Evidence: PicoSpec (2026,
[arxiv.org/abs/2603.19133](https://arxiv.org/abs/2603.19133)), real WAN,
Jetson↔A100, 1.13–2.90×, ablation isolating pipelining as the lever; DSD
(2026, [arxiv.org/abs/2511.21669](https://arxiv.org/abs/2511.21669)),
edge-cloud, explicit single-request-latency focus. Both are **research
prototypes** — 2025–26 preprints, real-hardware-plus-simulated-network
hybrids, not merged into any production runtime; vLLM's own docs
([docs.vllm.ai/en/latest/features/speculative_decoding/](https://docs.vllm.ai/en/latest/features/speculative_decoding/))
document no remote-drafter deployment mode at all. Token-tree/multi-drafter
verification on MoE specifically inflates cost: Cascade
([arxiv.org/abs/2506.20675](https://arxiv.org/abs/2506.20675)) shows
tree-draft verification widens the activated-expert union, inflating memory
traffic 2–3×, up to 1.5× slowdown; an independent RTX-3090 community
benchmark on Qwen3.6-A3B found 100% draft acceptance but every speculative
configuration still net-negative. Combined with this repo's own EAGLE3
failure, three independent signals converge negative for tree-verification
on this fine-grained, no-shared-expert MoE family — a zero-machine
architectural mismatch, separate from the async question.

**TP/PP/CP/KV/EP for batch-one and MoE.** Tensor parallelism is a
synchronous collective (all-reduce every layer); at batch=1 the message is
small and fixed but every layer still pays a sync barrier — Megatron-LM's
own guidance treats it as impractical once it crosses a node boundary
([arxiv.org/abs/2104.04473](https://arxiv.org/abs/2104.04473); **not
independently re-fetched** this pass, treat as plausible not confirmed, but
corroborated by DeepSpeed-Inference's own claim that "tensor slicing cannot
scale beyond a single node,"
[arxiv.org/abs/2207.00032](https://arxiv.org/abs/2207.00032)). Pipeline
parallelism ships activations point-to-point at stage boundaries; at
batch=1 the training bubble formula degenerates — pure additive per-token
latency, tolerant of heterogeneity but buying nothing. Sequence/context
parallelism (Megatron SP, ring attention, DeepSpeed Ulysses) shards the
sequence dimension and is inert at decode (seq_len=1/step) —
training/prefill-only. KV-parallel decode (Flash-Decoding, NVIDIA Helix,
[arxiv.org/abs/2507.07120](https://arxiv.org/abs/2507.07120)) is the true
decode-time analogue of CP: it shards the KV cache, combining via a
partial-output + log-sum-exp reduction whose volume is independent of
context length — lighter than TP's all-reduce, though NVIDIA's
32×-concurrency figure is a capacity claim, not single-stream evidence (its
1.5–1.8× min-latency figure is unreconciled between paper and blog). Expert
parallelism is P2P dispatch per routed expert — a slow worker delays only
its own routes, the most heterogeneity-tolerant of the four — but classic
EP-for-capacity is a **capacity** answer: LMSYS's own DeepSeek-EP report
([lmsys.org/blog/2025-05-05-large-scale-ep](https://lmsys.org/blog/2025-05-05-large-scale-ep),
confirmed verbatim) shows negative payoff below 64–128 tokens/device (−27%
at 32); batch=1 is 8 experts/layer from one token, far below every measured
breakeven. Qwen3-30B-A3B fits one GPU, so EP's capacity rationale is moot
here; multi-machine EP only helps for models too large for one GPU or for
cost-arbitrage against a weaker local device — never single-stream latency
at this scale. Maturity: TP/PP/EP are **mature/shipping**; KV-parallel
decode (Helix, vLLM DCP) is **very recent** (vLLM's blog dated 2026-08-07,
days old at review time).

**Remote/ensemble/multi-drafter.** SpecInfer
([arxiv.org/abs/2305.09781](https://arxiv.org/abs/2305.09781)), Sequoia
([arxiv.org/abs/2402.12374](https://arxiv.org/abs/2402.12374)), SpecTr
([arxiv.org/abs/2310.15141](https://arxiv.org/abs/2310.15141)) all boost
accepted-tokens/round via multiple draft candidates, but every one is
**colocated** (extra local GPUs on one host) — not evidence for "extra
machines over a network." Sequoia's own cited offload speedup on an L40 is
inconsistently reported across two specialists (9.5× vs. 10.33×, same
paper, unresolved) and should not be treated as load-bearing without a
fresh check. No source found anywhere in this review benchmarks a
genuinely distributed, heterogeneous ensemble of remote proposers for one
request — a literature gap, not a settled negative.

**Distributed verification, replicated lanes.** Full-replica hedging/racing
requires, per Shah/Lee/Ramchandran's queueing-optimality conditions
([arxiv.org/abs/1311.2851](https://arxiv.org/abs/1311.2851)), memoryless or
heavy-tailed service time *and* free cancellation for redundancy to help
mean latency; LLM decode has neither (regular per-step latency, non-zero
cancellation cost). This predicts redundancy helps only the tail below a
load threshold, and hurts throughput once concurrency rises — evidenced by
vLLM's own measured 1.4–1.8× **slowdown** at high QPS for the cheaper
multi-candidate-verification variant of "lanes"
([vllm-project.github.io/2024/10/17/spec-decode.html](https://vllm-project.github.io/2024/10/17/spec-decode.html)).
Dean & Barroso, "The Tail at Scale"
([dl.acm.org/doi/10.1145/2408776.2408794](https://dl.acm.org/doi/10.1145/2408776.2408794)),
defines hedged/tied requests and shows naive full redundancy without a
cutoff worsens tail. Non-batch-invariant kernels also mean racing replicas
aren't guaranteed to converge on the same output — "first to finish wins"
isn't distribution-neutral here the way it is for a stateless RPC race. No
primary source directly analyzes full-request replica racing for decode
specifically; this is triangulated, not a claim any cited paper makes.
**Rejected** as a direction.

**Early exit/cascade/future-token prediction.** Classic draft-model spec
decoding, n-gram/prompt-lookup, and CLLM/Jacobi decoding
([arxiv.org/abs/2305.10427](https://arxiv.org/abs/2305.10427),
[arxiv.org/abs/2403.00835](https://arxiv.org/abs/2403.00835), fixed point ≡
greedy AR, theorem-proven) are the only families both theorem-exact and
structurally offloadable as an independent component. EAGLE-family, Medusa,
both MTP variants, LayerSkip, CALM, and SkipDecode all require live target
hidden state on every step — architecturally **not offloadable**; CALM's
calibrated bound is explicitly non-exact and conflicts with this repo's
bit-exact mandate. DuoDecoding
([arxiv.org/abs/2503.00784](https://arxiv.org/abs/2503.00784), CPU+GPU same
box, 2.61×) and a multi-node InfiniBand result
([arxiv.org/abs/2511.11733](https://arxiv.org/abs/2511.11733), 2.56–2.59×)
are the only heterogeneous-hardware speedups found for this family, and
both need per-hop latency within 3–10× local compute to stay workable. No
primary source quotes p95/p99 for cross-machine spec decoding directly — a
literature gap, not a result.

**Heterogeneous Mac/cheap-GPU/CPU networking.** The only links with real
measured numbers here are Wi-Fi (71.6ms mean / 310.7ms max ICMP RTT —
disqualifying for per-layer dispatch at any percentile) and wired Mac LAN
Ethernet (15.3–19.9ms single-hop RPC; paired-block remote-boundary mean
105.7ms, fast mode 16–29ms, tail 216–298ms). Thunderbolt Bridge has **zero
primary latency figure anywhere** — the single largest measurement gap
here. RDMA-over-Thunderbolt-5 (Apple's JACCL,
[github.com/ml-explore/mlx/pull/2808](https://github.com/ml-explore/mlx/pull/2808),
confirmed real and merged) is official but ~2 months old at review time,
gated to M4 Pro/Max and M3 Ultra on macOS 26.2+, and its "order of
magnitude" latency claim is self-reported with no independent benchmark.
Consumer GPU P2P/GPUDirect is driver-disabled on GeForce 30/40-series;
cheap-GPU nodes always round-trip through host RAM. Same-AZ cloud Ethernet
is the only measured regime with real headroom for near-full-layer remote
dispatch (p95 1.330ms/hop); cross-AZ (p95 2.250ms/hop) is borderline once
the p95-sum-fallacy correction is applied. Maturity: gRPC/NCCL/InfiniBand/EFA
are mature but NCCL is NVIDIA-only, with no vendor-agnostic heterogeneous-GPU
collective support (HetCCL,
[arxiv.org/abs/2605.31000](https://arxiv.org/abs/2605.31000), third-party,
59–97% of NCCL bandwidth). MLX distributed (ring, now JACCL) is the only
**officially shipping**, non-experimental implementation of heterogeneous
batch-1 multi-node inference found in this review. llama.cpp's RPC backend
does the same mixed-hardware trick but is self-labeled by its own
maintainers "proof-of-concept... fragile and insecure."

**Scheduling, placement, correctness, and <100ms deadlines.** No mature
runtime audited (vLLM v0.27.1, SGLang v0.5.18, TensorRT-LLM/Dynamo v1.2.1,
DeepSpeed, Megatron-Core) implements heterogeneous-GPU-generation for one
request — vLLM issue
[#34437](https://github.com/vllm-project/vllm/issues/34437) documents real
crashes/silent numeric divergence from mixed RTX 3090/4090 GPUs (triggered
by `--pipeline-parallel-size 4` plus FP8-overflow between GPU generations),
closed stale with no fix. DeepSpeed's own team blog states ZeRO-Inference
(CPU/NVMe offload) is throughput/large-batch-only and unsuitable for
batch-1 — fetch latency can't be hidden without concurrent work to overlap
against. The feasible deadline/fallback architecture synthesized here (not
a claim any one paper makes) is: (1) prevent, don't react — bound per-step
latency structurally via chunked-prefill/stall-free scheduling so a large
concurrent prefill can't block the target stream's decode step
(Sarathi-Serve, [arxiv.org/abs/2403.02310](https://arxiv.org/abs/2403.02310),
OSDI'24, shipped in vLLM/SGLang); (2) treat speculation lookahead as a
load-adaptive lane-width knob, not full replication; (3) reactive fallback
only via live migration/priority promotion for at-risk streams (Llumnix,
[arxiv.org/abs/2406.03243](https://arxiv.org/abs/2406.03243), OSDI'24,
shipped, cuts tail latency "by an order of magnitude"), not a second full
replica; (4) hedge only the closer-to-idempotent prefill/TTFT step, never
full decode. No primary source ties a 100ms-ITL deadline to a fallback
policy — this is this review's own synthesis.

## Fundamental limits

- **Roofline/Amdahl ceiling.** T_token = Σ_l max(...) — a serial critical
  path across layers (design doc §9.1's normative formula, independently
  reproduced by two specialists). Remote compute is never faster than
  resident-VRAM compute for a model that already fits; every added hop is
  pure serial tax unless it unlocks genuine concurrency.
- **p95-sum fallacy**, corrected above: percentiles of a sum are not sums of
  percentiles; the 44/48-layer framing also silently swapped the SLA target
  from 16.6ms/token to 100ms/token.
- **No per-token ITL distribution exists in this repo today.** Only RPC-hop
  p50/p95 (n=20, no p99) and whole-forward mean/median (n=5) exist. Any
  p95/p99 <100ms claim for any candidate in this document is currently
  unverifiable.
- **Three correctness bars are conflated in the literature and must be kept
  separate**: exact-hash (this repo's actual bar, failed by all 3 tested
  variants), exact-distribution-in-theory (proven, not bit-exact on real
  kernels), and approximate/distributional (never licensed to substitute
  for the hash gate here).
- **Synchronous collectives cannot tolerate heterogeneity.** TP/CP require
  every member to synchronize every layer; a single slow or far node stalls
  the whole group, every layer, every token — this rules out the entire
  dense-parallelism-over-WAN class for this specific goal, independent of
  any other consideration.

## Ranked top three machine-causal research directions

Confidence and blockers stated per direction. **None of these is currently
demonstrated on this repo** — this ranking is a ranking of what to test next,
not a ranking of proven wins.

**1. Real decode-loop, sparse same-AZ remote-expert, KV-cache authoritative
(≤1–2 layers/token).** Mechanism: none positive is claimed — this *is* the
test for whether machines are causal at all, replacing today's
shadow-only/prefill-only harnesses with an authoritative remote critical
path plus real per-token instrumentation. Feasibility: **High** — infra
mostly exists (`full_model_shadow_eval.py` needs its remote path promoted
from shadow to authoritative). Hard blockers: no per-token percentile code
exists; n=5 blocks are underpowered. Confidence: **High** the test is
runnable now; **Low** confidence it shows a positive effect — the roofline
argument says remote compute has no speed edge over resident GPU compute,
so the likely honest outcome is "no worse within a sparse regime," an upper
bound on tolerable remote layers, not a throughput-scaling curve.

**2. Async remote draft/verify speculative decoding, genuinely pipelined
(PicoSpec/DSD-style).** Mechanism: the draft machine computes round i+1
while the target verifies round i — the only mechanism found anywhere in
this review where a second machine performs real concurrent work on the
same stream. Feasibility: **Low** — no async pipeline or speculative-KV
rollback exists in this repo or any mature runtime; MoE verification-cost
inflation (Cascade, confirmed) compounds the problem for this fine-grained,
no-shared-expert architecture. Hard blocker: the zero-RTT local proxy for
this mechanism — the standalone Qwen3-0.6B draft — already measured
**0.65×** (net slower) on the real GPU baseline; layering RTT on top of an
already-losing configuration is very unlikely to cross even. Confidence:
**Medium** the mechanism is real in general (PicoSpec/DSD show it on other
models); **Low** that it helps this specific model/stack soon.

**3. Same-AZ parallel multi-expert fetch as offload-and-cache (batch ~8
experts/layer into concurrent RPCs).** Mechanism: machines-causal only
versus a memory-constrained local baseline (CPU/NVMe offload —
Mixtral-Offloading, Fiddler, ktransformers, llama.cpp `--n-cpu-moe`), never
versus this repo's fast resident-VRAM GPU. Explicitly moot for
Qwen3-30B-A3B on the tested hardware, since the model already fits in VRAM.
Included only because it is the one place in the literature with a genuine
causal story for "more machines" — conditioned on hardware this repo
doesn't currently use as its comparison point. Feasibility: **Medium**
(single-expert RPC exists; batched dispatch does not). Confidence:
**Medium** for the mechanism in general; **not applicable** to the stated
goal on current hardware.

**Explicitly rejected:** full-replica hedging/racing (queueing theory +
vLLM's own slowdown data); classic EP-for-capacity (capacity ≠ throughput,
model doesn't need it); cross-AZ/multi-expert placement (roofline-
incompatible, statistically null at n=5); dense TP across heterogeneous
nodes (synchronous collective, a slower member provably worsens every layer
and token); token-tree/multi-drafter scaling on this MoE (Cascade + RTX-3090
community bench + this repo's own EAGLE3 failure converge negative,
independent of the machines question).

## Zero-machine prerequisite gates (must clear first, listed separately)

These are not scaling directions and were excluded from the ranking above
for that reason, even though several reviewers rated them individually
higher-priority than any machine-causal direction:

1. **EAGLE3 stable-wrong-hash root cause** — uncompiled rerun, batch-vs-solo
   logit diff, to isolate compile-path vs. tree-attention batch-invariance.
   Four specialists converged on this unprompted; grounded in Thinking
   Machines' confirmed batch-invariance mechanism. Gates whether
   spec-decoding can ever be a correctness-qualified lever here at all.
2. **Per-token ITL instrumentation** — no code path anywhere in this repo
   produces a real per-token p50/p95/p99 distribution today. Every
   p95/p99<100ms claim in this document is unverifiable without this.
3. **Wired Mac Ethernet/Thunderbolt baseline** — removes the Wi-Fi confound;
   the single highest-value missing native measurement, per this repo's own
   stated next gate.
4. **Statistical power** — n≫5 preregistered blocks before any placement CI
   is load-bearing.

## Staged experiment roadmap

| Stage | Machines | Content | Go/no-go gate |
|---|---|---|---|
| 0 | 0 | Wired Mac Ethernet/Thunderbolt re-run; confirm real `top_k`/experts-per-layer from the pinned checkpoint's `config.json`, not the illustrative YAML. | None — always run. |
| 1a | 0 | EAGLE3 uncompiled rerun + batched-vs-solo logit diff. | Exact-hash pass → spec-decoding is a real lever here; fail → deprioritize entirely, do not combine with any machine work. |
| 1b | 0 | Build real per-token p50/p95/p99 logging into the decode loop. | Must exist before any Stage 2 claim is valid. |
| 2 | 1+ | Direction #1: sparse same-AZ remote-expert, KV-cache authoritative, promote shadow path to authoritative, n≫5 preregistered blocks; same 1,000-token prompt/256-generated-token workload as the 2026-08-22 baseline, batch=1, exact-hash gate unchanged; per-token latency logged individually, report p50/p95/p99. | Remote-hop cost <10% of the 16.6ms token budget at p95, **and** the paired CI favors non-negative → proceed; else conclude placement is not causal at this model/scale and stop this line. |
| 3 | 1+ | Direction #3 (conditional): only if the program later targets a model/device that doesn't fit locally — batch 8 experts/layer into one RPC vs. llama.cpp `--n-cpu-moe`/ktransformers as a zero-network reference. | Beats the real local-offload baseline with real (not synthetic) compute. |
| 4 | 2+ | Direction #2 (conditional on 1a passing **and** a local-proxy re-test): only pursue async remote draft/verify if a local colocated draft first clears >1.0× on this exact model/GPU. | If the local proxy still loses, do not add RTT on top of a losing configuration. |

**Correctness definition carried through every stage**: exact full-sequence
output-hash equality against the single-machine reference, one output hash
per variant, matching the baseline hash — the same strict gate used in
`AWS-GENERATION-EXPERIMENT-2026-08-22.md`. Do not weaken it after seeing
results; a separate, explicitly-labeled quality/distributional experiment is
the only legitimate route to relaxing it.

**Deadline/fallback behavior for any future serving-shaped test**: prevent
via chunked-prefill/stall-free scheduling first (Sarathi-Serve), use
speculation lane-width as a load-adaptive knob (not full replication), and
reserve reactive migration/priority promotion (Llumnix-style) for genuinely
at-risk streams — never spin up a full replica of an ongoing decode to chase
a deadline. Hedge only prefill/TTFT if at all.

**Comparative baselines to carry into Stage 2+**: the existing single-GPU
SGLang floor (60.28 tok/s / 16.589ms/token / 85.9ms TTFT) as the zero-network
reference, and llama.cpp `--n-cpu-moe`/ktransformers as the zero-network
offload reference for Stage 3, so any measured number has a same-hardware,
zero-network comparison point rather than only a cross-machine one.

**Stopping criteria**: stop Stage 2 and downstream stages if (a) Stage 1a
fails — spec-decoding correctness is not fixable on this stack soon, so
don't compound it with network variance; (b) Stage 2's CI does not clear
zero after adequate power — conclude added machines are not causal for this
model at batch=1 and redirect the program to capacity/cost-arbitrage framing
instead of single-stream latency; (c) any stage's remote-hop p95 exceeds 10%
of the per-token budget — that stage's regime is disqualified regardless of
mean-case results, per the tail-compounding math above.

## Self-review

Sources: the adjudicator's final decision, all three critic handoffs
(critic-arch, critic-evidence, critic-quant), all ten specialist condensed
handoffs, and a direct re-read of both original-root ground-truth reports —
not specialist paraphrase alone. Every load-bearing number above (60.28
tok/s, 16.589ms/token, 85.9ms TTFT, 256 tokens, the four speculative-decoding
results, the near/far/local expert-boundary figures) was cross-checked
against the primary-source reports during this synthesis. Cited URLs are
primary papers or official docs/implementations the underlying reviews
already independently verified (arXiv abstracts fetched directly, GitHub
issues/PRs confirmed real and merged, vendor blogs confirmed verbatim), with
one flagged exception (Megatron-LM's TP-impractical-cross-node quote — PDF
fetch failed upstream, marked plausible-not-reconfirmed rather than
verified). Two unresolved cross-specialist contradictions — Sequoia's
offload speedup (9.5× vs. 10.33×, same paper) and Helix's 1.5×/1.8×
latency-improvement discrepancy (paper vs. NVIDIA blog) — are omitted from
this document's load-bearing claims rather than silently resolved. The
central conclusion (no repo evidence that added machines are causal for
single-stream throughput; none of the ranked directions is currently
demonstrated) is preserved unchanged from the adjudicator's finding.
