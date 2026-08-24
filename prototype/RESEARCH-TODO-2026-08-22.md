# Single-request heterogeneous scaling research TODO

Updated: 2026-08-24

## Production-fast baseline correction

- [x] Restore the production-fast target-only configuration: BF16, TP1,
  FlashInfer attention/sampling, CUDA graphs, no deterministic-inference flag,
  and no speculative algorithm.
- [x] Add cold-cache benchmarking by flushing SGLang's radix cache after the
  first warmup and before every later request. The confirmatory log records
  twelve 1,000-new-token/zero-cached-token requests.
- [x] Confirm the fixed 1,000-input/256-output batch-one contract on one G7e:
  168.25 pooled tok/s, 66.1 ms mean TTFT, 1.582 s mean end-to-end, and one stable
  output hash across ten measured repetitions.
- [x] Reclaim the run: OpenTofu destroyed all ten resources, followed by
  independent zero-instance and zero-EBS queries.
- [ ] Run the full agentic-coding quality suite on this production-fast target
  path. Target-only BF16 inference introduces no draft or approximation, but
  its token stream is not promised to be bit-identical to batch-invariant
  deterministic Triton at greedy near-ties.
- [ ] If causal attribution matters, run the remaining 2x2 backend/determinism
  screen. This is no longer required to meet 100 tok/s.

## Objective and fixed gates

Determine whether adding heterogeneous machines causally accelerates one
batch-one Qwen3-30B-A3B request while preserving the single-GPU reference
output and keeping client-visible inter-token-latency p95 and p99 below 100 ms.

- Planning reference: 60.28 tok/s, 16.589 ms/token mean, 85.9 ms TTFT.
  The confirmatory run-specific baseline was 59.036 tok/s, 16.939 ms/token,
  and 70.326 ms mean TTFT; all reported speedups and gates use that paired run.
- Correctness: one stable full-output token hash, equal to the reference hash.
- Primary latency estimand: remote/async minus local milliseconds per token;
  negative is better.
- Suggested minimum worthwhile improvement: 10% versus the reference. Confirm
  the final margin and sample size after a variance pilot, before confirmatory
  collection.
- Machine causality: report zero, one, two, and four *added* machines where the
  direction supports scaling. Local-only work never counts as a scaling win.

## Active work

- [x] Restore external prerequisites: `x.local` resolved to `192.168.1.153` and
  accepted SSH; `mi:scratchpad` SSO was renewed and the account has no live dAI
  EC2 instances or EBS volumes.
- [x] Add client-visible streaming event timestamps and p50/p95/p99 ITL to the
  generation harness, including explicit accounting for multi-token SSE chunks;
  unit tests cover both one-token and coalesced event streams.
- [x] Add request-aligned speculative acceptance ingestion using SGLang's native
  `/generate` response (`spec_accept_rate`, accepted/proposed drafts,
  verification count, and acceptance histogram); retain coarse server-log
  summaries as directional only. The pinned v0.5.16 OpenAI streaming adapter
  drops these fields, so a native diagnostic probe accompanies—not replaces—the
  client-visible streaming run.
- [x] Add an analysis tool for confidence intervals, equivalence/improvement
  gates, deadline-miss rate, and zero/one/two/four-machine scaling curves.

## Direction 1 — asynchronous remote draft/verify

- [x] Profile draft time for gamma={2,4,8,16} on the coordinator, Mac `x`, and
  a cheap AWS L4. The L4 eager loop exposed launch/synchronization overhead, so
  an optimized SGLang control was added: 166.37 tok/s, 6.010 ms mean ITL,
  6.190 ms p99, and 25.424 ms p99 TTFT. AWS CPU was gated out after two CPU
  classes and both eager GPU controls were slower than the target; another
  generic CPU cell could not change the decision.
- [x] Measure the remaining components: Wi-Fi remote-minus-local p50 was
  5.704 ms at 64 KiB; standalone request-aligned traces reported 155/404
  accepted/proposed draft tokens, 101 verification cycles, 2.535 output tokens
  per cycle, a 20/101 full-window rate, and a 62.54 ms observed cycle.
- [x] Evaluate the acceptance-aware PicoSpec-style screen using the measured
  acceptance histogram rather than treating aggregate acceptance as an
  independent per-token probability. The best measured component cell (L4
  SGLang plus Wi-Fi RTT and an optimistic one-baseline-step verification cost)
  is 16.65 ms/accepted token, only 1.017x baseline and below the 10% gate.
- [x] Gate out the target-authoritative one-drafter async prototype: no measured
  hardware/RTT cell predicts the required improvement. A zero-RTT model reaches
  1.14x, so sub-millisecond transport is a narrow follow-up hypothesis, not a
  measured scaling result.
- [x] Gate out two/four deadline-cancelled lanes because the one-drafter gate
  did not pass. The pinned SGLang release also has flags/protocol scaffolding
  but no completed remote-drafter execution path; upstream's current roadmap
  still lists verifier, drafter, sync-E2E, and overlap stages as unfinished.

Key hypothesis: useful accepted draft work overlaps target verification enough
to outweigh draft compute, transfer, synchronization, rejection, and rollback.

## Direction 2 — distributed multi-drafter/token-tree verification

- [x] Rerun EAGLE3 without compilation and compare output tokens. Uncompiled
  EAGLE3 is stable but first diverges from baseline at token 14; compiled EAGLE3
  first diverges at token 92. Compilation is therefore not the root cause.
  NGRAM and standalone also first diverge at token 14, implicating a shared
  speculative/batch-shape numerical path. A follow-up native target-logprob
  probe isolated the boundary: ordinary iterations use Triton decode attention,
  while `TARGET_VERIFY` uses Triton extend attention. Their BF16 logprobs differ
  immediately after prefill and flip a reported top-1 tie at token 14.
- [x] Capture request-aligned accepted tokens and verification counts before
  distribution: EAGLE3 121/402 and 134 cycles (1.910 output/cycle), compiled
  EAGLE3 115/426 and 142 cycles (1.803 output/cycle), standalone 155/404 and
  101 cycles (2.535 output/cycle). Tree-node, expert-union, and model-byte work
  was gated out because the prerequisite exact-output lock failed.
- [x] Gate out one/two/four remote drafters. No local speculative variant equals
  the reference hash, so distributing it would scale an invalid computation and
  could not support a machine-causal correctness-qualified claim.

Key hypothesis: independent remote candidates increase accepted tokens per
verification pass faster than MoE expert-union and tree-verification cost grows.

## Direction 3 — heterogeneous sharding/offload-and-cache

- [x] Apply the capacity gate. Qwen3-30B-A3B is resident on the 96 GB target,
  so a larger model or artificial VRAM cap would answer a different capacity
  question and cannot claim a latency win against this resident baseline.
- [x] Gate out zero-network CPU/NVMe offload references because the target does
  not require offload; they are only the correct comparator after a real
  memory-constrained target is selected.
- [x] Gate out layer-batched remote experts, hot replication, placement, and
  prefetch with one/two/four workers for the current benchmark. Reopen this
  direction only for a model that cannot remain resident.

Key hypothesis: only when weights cannot remain resident, coarse batched remote
memory/compute can beat the best local CPU/NVMe offload reference.

## Newly discovered issues and directions

- [x] LAN discovery issue: bare `x` does not resolve, but `x.local` resolves via
  mDNS. Inventory: Intel x86_64 Mac, macOS 12.7.6, 16 GiB RAM, Python 3.9.6.
- [x] AWS access issue: renewed the expired `mi:scratchpad` SSO token; verified
  no surviving dAI EC2 instances or EBS volumes before starting new work.
- [x] AWS capacity issue: CloudTrail confirmed repeated
  `Server.InsufficientInstanceCapacity` for `g7e.4xlarge` in `us-west-2`.
  Cancelled both retry-only launches, verified their complete cleanup, and
  relaunched in the previously proven `us-east-2`; it started in 25 seconds in
  `us-east-2b` while reading the protected west-region model cache.
- [x] Mac `x` initially had no Torch/Transformers/MLX runtime or cached Qwen
  model. Installed an isolated Python environment under `~/dai-draft-env` and
  began transferring the identical Qwen3-0.6B checkpoint for CPU profiling.
- [x] The coordinator has active 1 GbE on `en0`, while `x` exposes only active
  `en0` with no Ethernet media line and inactive `en1`; treat the current path as
  Wi-Fi. A wired-control claim was not made; confirming a wired adapter is a
  future environment change, not unfinished analysis.
- [x] Measure the actual `x` Wi-Fi transport component: remote-minus-local p50
  was 3.92 ms for 1 KiB, 5.70 ms for 64 KiB, and 9.85 ms for 64 KiB at
  concurrency eight; all responses were correct.
- [x] Profile local Qwen3-0.6B CPU draft compute: mean decode step was 24.56 ms
  (40.72 tok/s); gamma 2/4/8/16 rounds averaged 48.62/96.49/193.64/392.95 ms.
- [x] Profile the identical checkpoint on Intel Mac `x`: mean decode step was
  186.08 ms (5.37 tok/s); gamma 2/4/8/16 rounds averaged
  377.63/745.68/1503.10/2977.35 ms. This cell cannot meet the 100 ms/token
  target even before transfer and target verification, so it is a measured
  Direction-1 no-go rather than a prototype candidate.
- [x] Add and unit-test an acceptance-aware per-cycle screening tool. With the
  coarse 0.3353 acceptance value used only as a directional assumption and an
  optimistic one-baseline-step verification cost, the coordinator's best cell
  is 51.23 ms/token (0.324x baseline) and `x` is 297.62 ms/token (0.056x).
  Even at impossible perfect acceptance, the coordinator remains only
  0.69x baseline because its draft itself is slower than the target.
- [x] Stop and verify removal of the temporary transport service on `x` after
  collecting its measurements.
- [x] Streaming granularity risk handled: raw event timestamps are retained,
  exact token ITL is reported only for one-token events, and coalesced arrivals
  are flagged instead of receiving invented sub-event times.
- [x] Speculative divergence isolated: compilation, draft source/window,
  speculative attention-mode flag, and CUDA graphs are ruled out. The causal
  boundary is SGLang's numerically non-equivalent Triton decode-attention versus
  target-verify extend-attention paths; greedy `argmax` flips at a near-tie.
- [x] Current Qwen model fits the target GPU; Direction 3 was closed as not
  applicable rather than testing an irrelevant artificial regime.
- [x] L4 runtime issue: deterministic SGLang's persistent Triton matmul and its
  default prefill graph exceed L4 shared-memory limits. The speed-only control
  uses seeded ordinary kernels, disables prefill graphs, retains decode graphs,
  and is explicitly not used for correctness comparison.
- [x] AWS lifecycle: every successful, failed, and capacity-retry run completed
  teardown checks. Final status is zero live dAI instances and zero EBS volumes.

## Final disposition

- [x] Research plan complete. No approach achieved a correctness-qualified 10x
  result, and no direction passed its prerequisite gate for machine-count
  scaling. Future work is deliberately outside this completed run: obtain an
  exact-output speculative kernel and a production remote-drafter runtime, or
  select a genuinely non-resident model before reopening sharding/offload.

## Completed decisions

- [x] Literature/system review completed with ten specialists, three critics,
  adjudication, synthesis, and capped verification.
- [x] Async remote draft/verify selected as the sole currently credible
  machine-causal candidate.
- [x] Multi-drafter/tree verification retained as prototype-stage and
  evidence-adverse for this MoE until exactness and expert-union cost clear.
- [x] Sharding/offload retained only as a conditional capacity direction.
