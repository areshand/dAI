# AWS three-node placement experiment — 2026-08-20

## Result

The latency-aware planner correctly placed the two hot experts on the measured-near worker. In this synthetic single-request-stream workload, that placement cut mean block wall time by 51.8% relative to putting the hot experts on the far worker, and by 42.6% relative to the seeded random placement.

All four placements produced identical output digests. The experiment completed 10 randomized, counterbalanced blocks per policy (40 policy runs and 16,000 requests total).

| Policy | Hot experts | Mean wall time per 400 requests | Mean throughput | Median request latency |
|---|---|---:|---:|---:|
| Latency-aware | Near | 204.83 ms | 1,971.5 req/s | 0.267 ms |
| Hot-near oracle | Near | 206.68 ms | 1,944.5 req/s | 0.267 ms |
| Seeded random | Split | 356.79 ms | 1,124.4 req/s | 0.927 ms |
| Hot-far oracle | Far | 425.00 ms | 943.8 req/s | 0.996 ms |

The 1.84 ms difference between latency-aware and hot-near mean wall time is only 0.9%. They use the same placement, so this run should be read as parity rather than evidence that one is faster. Their small difference is ordinary run-order and host noise.

## Discovery signal

The coordinator sent 50 application-level probes with a 64 KiB payload to each worker before freezing placement:

| Worker | Topology | Probe p50 | Probe p95 | Worker compute p50 | Non-worker residual p50 |
|---|---|---:|---:|---:|---:|
| Near | Same AZ and cluster placement group | 0.211 ms | 0.456 ms | 0.038 ms | 0.173 ms |
| Far | Different AZ in the same region | 1.091 ms | 1.224 ms | 0.049 ms | 1.040 ms |

The far path's median application-level probe was 5.2× the near path. The planner ranked workers by measured end-to-end probe latency, not by a configured `near` or `far` label.

The worker-compute measurements are timings reported inside each worker for the same deterministic synthetic operation. They are recorded separately from coordinator-observed request latency. The 0.011 ms median difference between workers is not treated as network time or as a meaningful expert-speed result: at this scale it can reflect CPU scheduling, clock behavior, frequency, interrupts, and measurement overhead. The `non-worker residual` is calculated per probe as coordinator-observed latency minus worker-reported compute time; it includes network transfer plus coordinator/protocol overhead and is therefore not a pure wire-latency measurement.

## Method

- Three On-Demand `c7i.xlarge` Amazon Linux 2023 instances in `us-west-2`.
- Coordinator and near worker in one cluster placement group in one availability zone; far worker in another availability zone.
- Persistent TCP connections, 64 KiB payload, concurrency 1.
- Four synthetic experts; experts 0 and 1 receive most requests.
- 400 requests per policy per block, 10 blocks, seeded request stream and randomized policy order.
- Placement is frozen after discovery so each policy sees an explicit, reproducible mapping.
- No inbound SSH; orchestration used AWS Systems Manager.
- Every node had a verified independent three-hour termination timer. The local runner also destroyed the stack on exit.

## Cleanup

The run-scoped OpenTofu stack was destroyed immediately after result retrieval. A separate AWS inventory query found no live tagged instances or EBS volumes, and the OpenTofu state is empty.

## Claim boundary

This validates the measurement, discovery, placement-policy, counterbalancing, correctness, and cleanup mechanics of the testbed. It does **not** measure a real MoE model, GPU execution, expert weight transfer, batching, token-to-token routing, or quality. The next meaningful experiment is to replace the deterministic synthetic expert operation with an instrumented model runtime while retaining this same experimental control plane.
