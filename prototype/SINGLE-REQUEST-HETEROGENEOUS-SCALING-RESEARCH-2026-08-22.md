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
  far−near −2.10ms [-11.43, +5.58]. At n=5 this is underpowered and
  inconclusive, not a demonstrated win — and, per the roadmap below, also
  not proof that placement has no effect; no preregistered margin or power
  calculation was applied to this pilot-scale data.
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
reproducible divergence, not run noise. **This proves only that the
divergence is deterministic; it does not by itself prove why.** Two
competing, unresolved hypotheses exist: a compile-path/graph-capture effect
specific to this run, or Thinking Machines' documented batch-invariance
mechanism
([thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference)) —
matmul/RMSNorm/attention kernels are not batch-invariant, so a batched
tree-verification step *could* compute different logits than solo decode
for the same tokens. Neither is confirmed as the cause here; isolating them
is exactly what the planned uncompiled-rerun-plus-logit-diff experiment
(Stage 1a below) is for. **This correctness failure is not performance
evidence** — the EAGLE3 run was 1.59× *faster*, just wrong — so it should
not be read as a second signal reinforcing any performance conclusion (see
the Cascade discussion below, which is the independent MoE
verification-*performance* warning). N-gram's failure is a separate
mechanism — its draft length varies per run, so its batch shape (and
numerics) varies per run too. Two distinct root causes, not one bug — four
specialists converged on this independently.

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

The harness computes trial-level **p95 for TTFT, total (end-to-end)
latency, and output tokens/s** where it reports percentiles at all (e.g.
the AWS discovery probe's p50/p95 over n=20 calls). But **no experiment in
this repo has ever produced a per-token inter-token-latency (ITL)
distribution**, and **no p99 exists anywhere** — only p50/p95 (discovery
probe, n=20) or mean/median (whole-forward table, n=5).
`full_model_multi_worker_eval.py` runs with `use_cache=False` (prefill-only,
no decode loop); `full_model_shadow_eval.py` keeps its remote path
shadow-only, never authoritative for the actual output. No code path in
this repo produces the per-token number a `p95/p99 < 100ms` claim would
require. Any such claim made about any candidate below is currently
**unverifiable**, not merely unproven.

One specialist (hetero-net) computed: "same-AZ p95=1.330ms × 48 layers ≈
64ms," "cross-AZ p95=2.250ms × 48 ≈ 108ms → only 44/48 layers fit under
100ms." This is invalid percentile arithmetic: a sum of 48 independent
per-hop marginal p95 values has **no justified percentile label** for the
sum unless the joint distribution (or an explicit dependence/independence
model across hops) is specified — p95(Σxᵢ) is not, in general, Σp95(xᵢ),
and no such joint model was stated. A second, separate problem in the same
calculation: it silently swaps the SLA target from the measured
16.6ms/token baseline to a 100ms/token target — it "passes" only by
implicitly accepting decode collapsing from 60.28 tok/s to roughly 10 tok/s
(a 6× regression), a trade-off never stated as such. Any number built this
way should be read as an unlabeled, unjustified envelope, not a validated
latency percentile or a guarantee.

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
   Proven on paper, but **not guaranteed bit-exact** on real kernels that
   may not be batch-invariant — one candidate (unresolved) explanation for
   what this repo's EAGLE3 run demonstrates; see the EAGLE3 discussion
   above.
3. **Approximate/distributional quality** — Medusa's "typical acceptance"
   ([arxiv.org/abs/2401.10774](https://arxiv.org/abs/2401.10774)), CALM's
   conformal bound ([arxiv.org/abs/2207.07061](https://arxiv.org/abs/2207.07061)).
   Never validated here, and not licensed to substitute for bar (1) without a
   separate quality experiment — which the source report itself declines to
   run, warning explicitly against weakening the timing gate after results.

## Topic-by-topic literature map, with maturity labels

**Async/disaggregated speculative decoding, token-tree verification.** The
mechanism that lets a *second machine* do useful concurrent work on one
stream is pipelining: naive synchronous remote drafting pays the full
serial cost `L_sync = t_draft + RTT + t_verify` and is a net loss unless the
target is already slow. llama.cpp issue
[#23982](https://github.com/ggml-org/llama.cpp/issues/23982) (closed,
unimplemented proposal for a distributed draft/target proxy) states this
threshold explicitly: "network overhead is <1% of total latency for
targets slower than ~50 t/s." Genuinely asynchronous pipelining does
**not** simply replace this with `max(t_draft, RTT+t_verify)` — that is
only the full-hit case where every drafted token is accepted. PicoSpec's
own probabilistic model
([arxiv.org/abs/2603.19133](https://arxiv.org/abs/2603.19133) §3.4, Eq.
8–9) gives the expected per-round cost, acceptance-aware, as `E[T] = α^γ ·
max(t_draft, RTT+t_verify) + (1 − α^γ) · (t_draft + RTT + t_verify)`, where
γ is the drafted-token window and α the (assumed stationary, independent)
per-token acceptance rate: with probability α^γ every token in the window
is accepted (a "full hit," amortized at the max(...) rate); otherwise
(probability 1−α^γ) a rejection forces a flush/resync, paying the full
synchronous cost. This degrades gracefully — as α→0, E[T] approaches the
synchronous baseline, never worse — but it is an expectation over many
rounds under an idealized independent-acceptance model, not the bare
max(...) figure. **This repo has not logged a compatible α** for the
standalone Qwen3-0.6B/Qwen3-30B-A3B draft-target pair — its 0.65×
colocated result reports only aggregate decode throughput, not a per-token
acceptance rate — so this formula cannot yet be evaluated on this repo's
own model pair. PicoSpec itself reports 1.13–2.90× speedup across its four
model/dataset cells (Qwen 0.6B/32B and Llama 1B/70B on GSM8K/HumanEval,
real WAN, Jetson↔A100) — a **research prototype**, a 2026 preprint, not
merged into any production runtime reviewed here. Its own Qwen-GSM8K main
cell ("Ours (Full)," Table 2) reports a **mean TPOT of 148.84ms**
(T_draft≈97ms, T_verify≈87ms) and **no p95/p99 anywhere in the paper**;
that cell's mean already exceeds this task's 100ms target on its own, and
with no tail data reported, it does not establish this task's under-100ms
tail objective either way. (arXiv:2511.21669,
"DSD," is discrete-event fleet serving/routing/batching evidence and is
**not** batch-one async-pipelining latency evidence — it was miscited as
such in an earlier pass of this synthesis and is removed from this claim.)
vLLM's own docs
([docs.vllm.ai/en/latest/features/speculative_decoding/](https://docs.vllm.ai/en/latest/features/speculative_decoding/))
document no remote-drafter deployment mode at all in that runtime.
Separately, this repo's colocated standalone Qwen3-0.6B draft measured
**0.65×** (net slower) sharing the *same* target GPU with the target model —
that shared-resource contention is a warning sign about draft/verify
overhead, but it is **not a zero-RTT proxy for, or a hard blocker on,
genuine disaggregation**: moving the draft to independent compute/memory
bandwidth is a structurally different resource condition that could enable
real overlap a colocated, resource-sharing draft cannot. Judging the async
direction requires component-level timing (t_draft, t_verify, acceptance
rate α, measured separately) plus the acceptance-aware model above, not an
inference from the colocated number. Token-tree/multi-drafter verification
on MoE specifically inflates cost: Cascade
([arxiv.org/abs/2506.20675](https://arxiv.org/abs/2506.20675)) shows
tree-draft verification widens the activated-expert union, inflating memory
traffic 2–3×, up to 1.5× slowdown — a performance warning, independent of
this repo's own EAGLE3 correctness failure (see the EAGLE3 discussion
above: a *stable-wrong-hash* result is evidence of deterministic
divergence, not of a performance problem — EAGLE3's raw run was faster,
just wrong — so it is not combined with Cascade here as a second
convergent performance signal). This is a zero-machine architectural
mismatch, separate from the async question. (A third, informal RTX-3090
community benchmark on Qwen3.6-A3B was cited in an earlier pass; it is a
non-peer-reviewed community report and is treated here as non-load-bearing.)

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
decoding, n-gram/prompt-lookup, and training-free Jacobi decoding
([arxiv.org/abs/2305.10427](https://arxiv.org/abs/2305.10427), fixed point
≡ greedy AR for that same model, theorem-proven) are the only members of
this family that are both theorem-exact and structurally offloadable as an
independent component. CLLM
([arxiv.org/abs/2403.00835](https://arxiv.org/abs/2403.00835)) is a
**different guarantee**, not the same bar: it requires fine-tuning/distilling
the target model to consistently predict its own fixed point, and its own
abstract claims only "preserving generation quality" — not bit-exact
identity to the original, un-tuned target. Reclassified here as
quality-preserving/fine-tuned rather than bit-exact, absent primary
evidence otherwise. EAGLE-family, Medusa, both MTP variants, LayerSkip,
CALM, and SkipDecode all require live target hidden state on every step —
architecturally **not offloadable**; CALM's calibrated bound is explicitly
non-exact and conflicts with this repo's bit-exact mandate. DuoDecoding
([arxiv.org/abs/2503.00784](https://arxiv.org/abs/2503.00784), CPU+GPU same
box, 2.61×) is a genuine heterogeneous-hardware speedup for this family,
needing per-hop latency within 3–10× local compute to stay workable. (A
separately-numbered paper, [arxiv.org/abs/2511.11733](https://arxiv.org/abs/2511.11733)
— also a "DSD," distinct from arXiv:2511.21669 discussed above — reports
2.56–2.59× but is **not** heterogeneous-hardware evidence: its own §3.1
states every node runs one identical NVIDIA A800 over a 100Gbps InfiniBand
link, and its node-scaling ablation ("we simulate deployments with two to
sixteen nodes") is explicitly simulated, not measured on real multi-node
hardware. Its best-performing variant also relaxes exact target-distribution
matching for non-key tokens via a softened distribution (τ up to 0.3, its
own Eq. 8) — an approximate, quality-preserving method that does **not**
satisfy this repo's exact-hash bar. It is removed from both the
heterogeneous-hardware and exact-match claims here.) No primary source
quotes p95/p99 for cross-machine spec decoding directly — a literature
gap, not a result.

**Heterogeneous Mac/cheap-GPU/CPU networking.** The repo's own Mac
measurements are **not** a wired baseline: the coordinator reached the
worker over the coordinator's `en0` **Ethernet** interface while the worker
itself was on `en0` **Wi-Fi** (`EXPERIMENT-2026-08-20.md`) — a mixed,
asymmetric link, not a wired-both-ends control. On that topology: 50 ICMP
samples averaged 71.6ms mean / 310.7ms max RTT; the paired remote-expert
boundary averaged 105.7ms (fast cells 16–29ms, tail 216–298ms). The report
itself states the ICMP tail matches the RPC tail — the Wi-Fi leg, not
compute or serialization, dominates — and explicitly flags that "a wired
Ethernet or Thunderbolt link... is required before estimating the minimum
local-placement overhead," which this repo has not yet run. Thunderbolt
Bridge has **zero primary latency figure anywhere** — the single largest
measurement gap here. RDMA-over-Thunderbolt-5 (Apple's JACCL,
[github.com/ml-explore/mlx/pull/2808](https://github.com/ml-explore/mlx/pull/2808),
merged 2025-12-17, ~8 months before this review) is officially shipped but
explicitly **experimental**: MLX's own docs
([ml-explore.github.io/mlx/.../usage/distributed.html](https://ml-explore.github.io/mlx/build/html/usage/distributed.html))
state "until the feature matures, enabling RDMA over thunderbolt is
slightly more involved and cannot be done remotely even with sudo" (it
requires macOS recovery-mode setup) and that the backend "supports only
fully connected topologies" (full-mesh cabling only; ring/bandwidth
optimization "should come later" per the PR itself), gated to macOS 26.2+
and Thunderbolt-5 Macs (M3 Ultra/M4 Pro-Max in Apple's own examples). Its
"order of magnitude lower latency than ring" claim is self-reported with no
independent benchmark found in this review. NVIDIA's own GPUDirect RDMA
documentation states it is "available on both Tesla and Quadro GPUs"
([docs.nvidia.com/cuda/gpudirect-rdma/](https://docs.nvidia.com/cuda/gpudirect-rdma/))
without listing GeForce — this shows the officially documented support
excludes GeForce, but it does not prove every consumer-GPU topology must
round-trip through host RAM; no primary source in this review measures
consumer-GPU P2P directly. Same-AZ cloud Ethernet and cross-AZ cloud
Ethernet both have real per-hop p95 measurements (1.330ms and 2.250ms
respectively); whether either regime leaves real end-to-end headroom under
the 100ms target is **not established** — that would require the
per-token joint measurement described above, not a sum of marginal p95s.
Nominal arithmetic only: even taking the (invalid) 48-layer sum at face
value, ≈64ms (same-AZ) would already be a severe regression from the
16.589ms/token baseline (~4× slower) — well before asking whether it clears
the looser 100ms cap — and ≈108ms (cross-AZ) would be worse still; no
end-to-end percentile conclusion follows from this arithmetic without a
joint model and an actual per-token measurement. Maturity: gRPC/NCCL/InfiniBand/EFA are mature but NCCL is
NVIDIA-only, with no vendor-agnostic heterogeneous-GPU collective support
found in this review (HetCCL,
[arxiv.org/abs/2605.31000](https://arxiv.org/abs/2605.31000), third-party,
59–97% of NCCL bandwidth). Among the runtimes reviewed, MLX's ring backend
is the most mature shipping implementation of heterogeneous batch-1
multi-node inference; its JACCL/RDMA path is real and merged but
maintainer-labeled experimental, as above. llama.cpp's RPC backend does a
similar mixed-hardware trick but is self-labeled by its own maintainers
"proof-of-concept... fragile and insecure."

**Scheduling, placement, correctness, and <100ms deadlines.** No mature
runtime audited (vLLM v0.27.1, SGLang v0.5.18, TensorRT-LLM/Dynamo v1.2.1,
DeepSpeed, Megatron-Core) implements heterogeneous-GPU-generation for one
request — vLLM issue
[#34437](https://github.com/vllm-project/vllm/issues/34437) documents real
crashes/silent numeric divergence from mixed RTX 3090/4090 GPUs (triggered
by `--pipeline-parallel-size 4` plus FP8-overflow between GPU generations),
closed stale with no fix. DeepSpeed's own blog
([deepspeed.ai/2022/09/09/zero-inference.html](https://www.deepspeed.ai/2022/09/09/zero-inference.html))
states ZeRO-Inference (CPU/NVMe offload) "is optimized for inference
applications that are throughput-oriented and allow large batch sizes,"
while alternatives are "more suitable for inference applications that are
latency sensitive or have small batch sizes" — i.e. unsuitable for batch-1
by the vendor's own framing; fetch latency can't be hidden without
concurrent work to overlap against. The feasible deadline/fallback
architecture synthesized here (not
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
- **p95-sum fallacy**, corrected above: a sum of marginal per-hop p95
  values has no justified percentile label for the sum without a stated
  joint/dependence model; the 44/48-layer framing also silently swapped
  the SLA target from 16.6ms/token to 100ms/token.
- **No per-token ITL distribution exists in this repo today.** The harness
  reports trial-level p95 for TTFT/total latency/output TPS where it
  reports percentiles at all; only the RPC-hop discovery probe (n=20, p50/p95,
  no p99) and the whole-forward table (n=5, mean/median) exist otherwise.
  Any per-token p95/p99 <100ms claim for any candidate in this document is
  currently unverifiable.
- **A CI crossing zero is inconclusive, not proof of non-causality.**
  Without a preregistered minimum-worthwhile-effect margin and a power
  calculation behind it, an underpowered zero-crossing interval licenses
  neither "this works" nor "added machines don't help" — only "collect
  more data, or state an equivalence margin and test against it."
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

**Stated plainly: the literature and this repo's evidence do not currently
yield three credible machine-causal scaling directions for this goal.**
Only one candidate has a plausible mechanism and a non-negative evidence
trajectory. The other two "top-3" slots below are included to satisfy the
requested ranking format — one is prototype-stage and evidence-adverse for
this specific MoE, the other is conditional on hardware this repo doesn't
use. **None of the three is currently demonstrated on this repo.**

**1. Truly async remote draft/verify speculative decoding, genuinely
pipelined (PicoSpec-style) — the sole credible candidate today.**
Mechanism: the draft machine computes round i+1 while the target verifies
round i — the only mechanism found anywhere in this review where a second
machine does real concurrent work on the same stream, since disaggregation
gives the draft independent compute/memory bandwidth a colocated draft
cannot access. The expected per-round cost is **acceptance-aware**, not a
bare max(...) figure — see the async topic note above for PicoSpec's own
formula and its Qwen-GSM8K cell (148.84ms mean TPOT, already over this
task's 100ms target, no p95/p99 reported). Feasibility: **Low** — no async
pipeline or speculative-KV rollback exists in this repo or in any of the
mature runtimes reviewed; MoE verification-cost inflation (Cascade,
confirmed) compounds the problem here. The colocated standalone
Qwen3-0.6B draft's **0.65×** (net slower, sharing the target GPU) is a
warning about draft/verify overhead, **not** a zero-RTT proxy for, or a
hard blocker on, disaggregated operation (see the async topic note above).
Required before this can be ranked with confidence — the Direction 1
pipeline, Stages 1b→1c→1d→4 below: real per-token ITL instrumentation,
then measured t_draft/t_verify/α for this repo's own draft-target pair
(not inferred from the shared-GPU aggregate), then the acceptance-aware
model evaluated at candidate RTTs, before any actual two-node test.
Confidence: **Medium** the mechanism is real in general (PicoSpec shows
genuine async pipelining beating baseline over real WAN on other models);
**Low/unknown** for this specific model/stack, pending that
measurement-and-modeling pipeline.

**2. Distributed multi-drafter / token-tree verification — prototype-stage,
low confidence, likely adverse for this Qwen MoE.** Mechanism: extend
tree-based multi-candidate verification (SpecInfer/Sequoia/SpecTr-style) to
genuinely distributed drafters rather than colocated extra GPUs. No source
found in this review benchmarks that distributed configuration over a
network — a literature gap, not a settled result. Feasibility: **Low**.
Evidence specific to this MoE is negative on two distinct, non-redundant
axes, both zero-machine and independent of the distribution question:
*performance* — Cascade's confirmed 2–3× verification memory-traffic
inflation for tree-based drafts predicts underperformance from the extra
compute/memory traffic this MoE's fine-grained, no-shared-expert routing
creates; and *correctness* — this repo's own EAGLE3 stable-wrong-hash
failure shows tree verification can deterministically diverge from the
target on this stack (root cause unresolved — see above), separately from
whether it would otherwise be fast. The EAGLE3 result is not itself
performance evidence (that run was 1.59× faster, just wrong) and is not
combined with Cascade as a second convergent slowdown signal — each is an
independent reason to deprioritize Direction 2. Confidence: **Low**.

**3. Heterogeneous sharding / offload-and-cache (same-AZ parallel
multi-expert fetch into concurrent RPCs, or CPU/NVMe offload) — conditional
only.** Mechanism: machines-causal *only* versus a memory-constrained local
baseline (a model/device that cannot hold the full model), never versus
this repo's fast resident-VRAM GPU. **Explicitly not applicable** to
Qwen3-30B-A3B on the tested hardware, since the model already fits in one
GPU's VRAM. Included to satisfy the requested top-3 format, scoped strictly
as conditional on a future model or device too large/weak to hold the
model locally (see llama.cpp `--n-cpu-moe`, ktransformers, Mixtral-Offloading,
Fiddler for the relevant zero-network offload baselines). Feasibility:
**Medium** (single-expert RPC exists; batched multi-expert dispatch does
not). Confidence: **Medium** for the mechanism in general on a smaller/weaker
device; **not applicable** to the stated goal on current hardware.

**Diagnostic/boundary experiment, not a scaling direction:** sparse
same-AZ remote-expert decode with KV-cache authoritative (≤1–2
layers/token). This claims **no positive mechanism** — it is the
boundary/diagnostic test that characterizes whether a remote expert call is
even tolerable within the per-token time budget, which in turn informs
Directions 1 and 3 above. It is deliberately not ranked as a candidate
because, per the roofline argument, remote compute has no speed edge over
resident-GPU compute for a model that fits — the honest best case is "no
worse within a sparse regime," an upper bound on tolerable remote layers,
not a throughput-scaling curve. See Stage 2 of the roadmap below.

**Explicitly rejected:** full-replica hedging/racing (queueing theory +
vLLM's own slowdown data); classic EP-for-capacity (capacity ≠ throughput,
model doesn't need it); cross-AZ/multi-expert placement (roofline-
incompatible, statistically underpowered at n=5 — see the roadmap's note on
not inferring non-causality from an unpowered CI); dense TP across
heterogeneous nodes (synchronous collective, a slower member provably
worsens every layer and token).

## Zero-machine prerequisite gates (must clear first, listed separately)

These are not scaling directions and were excluded from the ranking above
for that reason, even though several reviewers rated them individually
higher-priority than any machine-causal direction:

1. **EAGLE3 stable-wrong-hash root cause** — uncompiled rerun, batch-vs-solo
   logit diff, to isolate compile-path vs. tree-attention batch-invariance —
   two competing, unresolved hypotheses (see the EAGLE3 discussion above);
   this stable-wrong-hash result only proves the divergence is
   deterministic, not why. Four specialists converged on running this
   isolation experiment unprompted. Gates whether tree-verification-family
   spec-decoding (Direction 2) can ever be a correctness-qualified lever
   here — kept independent of the remote-expert and standalone-draft
   tracks below (different mechanisms, different failure modes).
2. **Per-token ITL instrumentation** — no code path anywhere in this repo
   produces a real per-token p50/p95/p99 distribution today; the harness
   only reports p95 for trial-level TTFT/total-latency/output-TPS. Every
   per-token p95/p99<100ms claim in this document is unverifiable without
   this. (= Stage 1b below.)
3. **Genuinely wired (both ends) Mac Ethernet/Thunderbolt baseline** — the
   existing Mac numbers are coordinator-Ethernet-to-worker-Wi-Fi, a mixed
   asymmetric link, not a wired control (`EXPERIMENT-2026-08-20.md`); this
   repo has not yet run a true wired-both-ends cell. (= Stage 0 below.)
4. **Measured draft/verify/acceptance traces, then an acceptance-aware
   overlap model** — t_draft, t_verify, and per-token acceptance rate α for
   the standalone/classic draft-target mechanism (not EAGLE), measured
   colocated, then fed into PicoSpec's acceptance-aware cost formula (see
   the async topic note above) to predict whether Direction 1 is worth an
   actual two-node test. (= Stages 1c and 1d below.)
5. **Statistical design** — a pilot run to estimate variance, followed by an
   explicit power calculation for a preregistered minimum-worthwhile-effect
   or equivalence margin, replacing any arbitrary large-n rule before a
   placement or draft/verify CI is treated as load-bearing.

## Staged experiment roadmap

**Estimand and sign convention, fixed once for every stage:** define
Δ_lat = latency(remote/async) − latency(local), in ms/token (or ms for
TTFT). Improvement (remote/async faster) means **Δ_lat < 0**. Fix a
minimum-worthwhile-improvement δ>0 (ms) and an equivalence half-width ε>0
(ms) *before* collecting data. **Improvement gate**: the CI's *upper*
bound for Δ_lat is ≤ −δ. **Equivalence gate**: the *entire* CI for Δ_lat
lies within [−ε, +ε]. A CI that clears neither gate is **inconclusive**
(not "not causal") — collect more data per the power calculation. Where a
stage's natural metric is throughput instead, define Δ_thr =
throughput(remote/async) − throughput(local), tok/s, with the **opposite**
sign logic: improvement means Δ_thr > 0 (CI *lower* bound ≥ +δ_thr);
equivalence means the entire CI lies within [−ε_thr, +ε_thr]. Every stage
below states which estimand it uses. All use a small pilot to estimate
variance, then an explicit power calculation for the preregistered margin —
not an arbitrary "n≫5" rule. "Machines" below counts **added** machines
relative to the single-GPU baseline that already exists.

**Direction 1's own pipeline** (the only prerequisite chain that gates
Direction 1) is 1b → 1c → 1d → 4: build instrumentation, measure real
draft/verify/acceptance traces, build the acceptance-aware model, only
then run the actual two-node async test. Stage 0 (wired-Mac
characterization), Stage 1a (EAGLE), and Stage 2 (remote-expert diagnostic)
are **parallel side tracks** — useful in their own right, but none of them
gates or is gated by Direction 1.

| Stage | Added machines | Content | Go/no-go gate |
|---|---|---|---|
| 0 (side track) | 1 | Genuinely wired-both-ends Mac Ethernet/Thunderbolt run (distinct from the existing coordinator-Ethernet/worker-Wi-Fi data); confirm real `top_k`/experts-per-layer from `config.json`. | None — always run. |
| 1a (EAGLE side track) | 0 | EAGLE3 uncompiled rerun + batched-vs-solo logit diff. | Exact-hash pass → Direction 2 (tree-verification) is a real lever; fail → deprioritize Direction 2 only. Does not gate Direction 1 or Stage 2. |
| 1b (Direction 1, step 1) | 0 | Build real per-token p50/p95/p99 ITL logging into the decode loop. | Must exist before any per-token percentile claim at Stage 2 or 4 is valid. |
| 1c (Direction 1, step 2) | 0 (colocated) | Measure t_draft, t_verify, and per-token acceptance rate α separately for the standalone/classic draft-target mechanism (real traces, not assumed). | Feeds Stage 1d; a losing colocated aggregate number does not by itself block the pipeline (see the async topic note on why colocated ≠ disaggregated). |
| 1d (Direction 1, step 3 — modeling only) | 0 | Using 1c's measured t_draft, t_verify, α and the acceptance-aware formula (`E[T] = α^γ·max(t_draft,RTT+t_verify) + (1−α^γ)·(t_draft+RTT+t_verify)`), compute predicted E[T] across candidate RTTs and compare to the resident single-GPU baseline. | Predicted Δ_lat's sign and magnitude at the target RTT set the δ Stage 4 must clear; if no candidate RTT predicts Δ_lat < 0, that is informative and Stage 4 should not proceed yet. |
| 2 (diagnostic/boundary side track) | 1 | Sparse same-AZ remote-expert, KV-cache authoritative; same 2026-08-22 workload/exact-hash gate; pilot for variance, then power-calculated n; per-token p50/p95/p99. Estimand: Δ_lat. | End-to-end visible-ITL p95/p99 **<100ms**, and Δ_lat's CI upper bound ≤ −δ (declare improvement, informs Direction 3) or entire CI within [−ε,+ε] (declare equivalence) — a CI clearing neither means collect more data, not "not causal." |
| 3 (Direction 3, conditional) | 1+ | Only if a future model/device doesn't fit locally: batch experts/layer into one RPC vs. llama.cpp `--n-cpu-moe`/ktransformers zero-network reference. States up front whether it uses Δ_lat or Δ_thr; same gate logic as Stage 2 with that estimand's sign. | Clears the improvement gate for its declared estimand; end-to-end p95/p99 <100ms where a latency SLA applies. |
| 4 (Direction 1, step 4) | 1 | Only after Stage 1d predicts a plausible win at the target RTT: run the actual async disaggregated pipeline (not the colocated proxy), same exact-hash gate. Estimand: Δ_lat. | End-to-end visible-ITL p95/p99 **<100ms**, and Δ_lat's CI upper bound ≤ −δ. |

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

**Stopping criteria**: (a) Stage 1a failing stops/deprioritizes Direction 2
(tree-verification spec-decoding) specifically — it does **not** stop the
Direction 1 pipeline (1b→1c→1d→4) or Stage 2, which test different
mechanisms and have their own correctness gates. (b) For any stage, if the
pilot-informed power calculation shows the study cannot yet distinguish the
preregistered minimum-worthwhile effect from zero (i.e. clears neither the
improvement nor the equivalence gate above), the correct action is to
collect more data per that calculation — not to declare "added machines
are not causal." Declare "no meaningful effect" only when a fully-powered
CI falls inside the predefined equivalence margin. (c) Any stage whose
end-to-end visible-ITL p95/p99 exceeds 100ms disqualifies that regime
outright, replacing any flat per-hop threshold.

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
single-stream throughput) is preserved unchanged from the adjudicator's
finding.

**Verifier iteration-1 corrections applied**, each re-verified directly
this pass (not taken on trust): Mac topology corrected to
coordinator-Ethernet/worker-Wi-Fi, not wired (`EXPERIMENT-2026-08-20.md`);
MLX JACCL PR [#2808](https://github.com/ml-explore/mlx/pull/2808) merge
date corrected to 2025-12-17 (~8 months, via `gh pr view`) and labeled
experimental per its own text and MLX's official docs (fetched directly);
llama.cpp issue [#23982](https://github.com/ggml-org/llama.cpp/issues/23982)
added as a direct link (via `gh issue view`: closed/unimplemented, confirms
the <1%-below-50-tok/s threshold verbatim); DeepSpeed's ZeRO-Inference blog
([deepspeed.ai/2022/09/09/zero-inference.html](https://www.deepspeed.ai/2022/09/09/zero-inference.html))
and NVIDIA's GPUDirect RDMA docs
([docs.nvidia.com/cuda/gpudirect-rdma/](https://docs.nvidia.com/cuda/gpudirect-rdma/))
added as direct links, both fetched and confirmed verbatim. The ranking was
reframed to one credible direction (async draft/verify) plus two
relabeled prototype/conditional slots; RTX-3090 and DSD (arXiv:2511.21669)
were removed as batch-one evidence; the 0.65× colocated result was
reframed from "hard blocker" to "warning requiring component timing"; the
invalid 0.95⁴⁸ argument was removed; and roadmap gates now use
preregistered estimands, equivalence margins, and power calculations
instead of a flat n≫5 rule or an arbitrary per-hop percentage.

**Verifier iteration-2 corrections applied**, each re-verified directly
(PicoSpec's full PDF read via `Read`; arXiv:2511.11733's title confirmed
via search and its full PDF read via `Read` to disambiguate it from
arXiv:2511.21669, whose own title/abstract were re-confirmed via search;
CLLM's abstract fetched): the async-decoding cost model is now acceptance-aware
(PicoSpec Eq. 8–9), not a bare max(...), with PicoSpec's own Qwen-GSM8K
148.84ms-mean-TPOT/no-p95-p99 cell disclosed; the estimand sign is now
fixed and unambiguous (Δ_lat/Δ_thr, opposite-sign conventions, explicit
improvement/equivalence gates) across Stages 2–4; the percentile section no
longer claims real end-to-end headroom or cross-AZ "borderline feasibility"
from summed marginal p95s, stating only nominal arithmetic and the implied
regression from baseline; arXiv:2511.11733 is reclassified as homogeneous-hardware
(A800/100Gbps InfiniBand, per its own §3.1), simulated node-scaling, and
relaxed (non-exact) verification — not heterogeneous or exact-hash
evidence; CLLM is reclassified as quality-preserving/fine-tuned, not
bit-exact to the original target; EAGLE3's stable-wrong-hash result is now
treated as proof of deterministic divergence only (compile-path vs.
batch-invariance left as competing unresolved hypotheses) and is no longer
combined with Cascade as convergent performance evidence; the GeForce/GPUDirect
sentence no longer claims every consumer topology must round-trip through
host RAM; and the roadmap now states explicitly that only Stages
1b→1c→1d→4 gate Direction 1, with Stage 0/1a/2 as parallel side tracks, and
labels the "Machines" column as machines added beyond the existing
single-GPU baseline.
