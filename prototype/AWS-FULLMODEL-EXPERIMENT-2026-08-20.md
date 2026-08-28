# AWS real full-model placement experiment — 2026-08-20

## Result

The real-expert discovery selected the same-AZ `near` worker. All 15 measured full Qwen forwards preserved the original routes and greedy token, and every final-logit tensor was bit-for-bit equal to the local reference.

The near expert boundary was consistently faster than the cross-AZ boundary, but five counterbalanced blocks do not establish a whole-forward performance difference: every paired 95% interval crossed zero.

## Real-expert discovery

The coordinator sent 20 real seven-row BF16 expert calls to each warmed worker before freezing the selection:

| Worker | Topology | Boundary p50 | Boundary p95 | Worker compute p50 | Unattributed p50 |
|---|---|---:|---:|---:|---:|
| Near | Same AZ and cluster placement group | 0.835 ms | 1.330 ms | 0.390 ms | 0.321 ms |
| Far | Different AZ in the same region | 2.160 ms | 2.250 ms | 1.040 ms | 0.945 ms |

The far boundary median was 2.6× the near boundary median. This is an application-service measurement, not pure wire latency: both worker compute and the residual transport/protocol component favored the near worker.

## Measured full forwards

Each path ran once for warmup and five times in randomized three-path blocks. Expert 53 was naturally selected once per forward for seven prompt rows.

| Path | Mean full forward | Median full forward | Mean expert boundary | Mean worker compute |
|---|---:|---:|---:|---:|
| Local | 1,078.84 ms | 1,079.23 ms | — | — |
| Near | 1,083.72 ms | 1,082.11 ms | 2.019 ms | 1.048 ms |
| Far | 1,081.63 ms | 1,083.42 ms | 3.183 ms | 1.629 ms |

| Paired comparison | Mean delta | Paired-bootstrap 95% interval |
|---|---:|---:|
| Near minus local | +4.88 ms | -0.66 to +11.08 ms |
| Far minus local | +2.78 ms | -1.23 to +6.79 ms |
| Far minus near | -2.10 ms | -11.43 to +5.58 ms |

The direct far boundary was 1.16 ms slower on average, yet the far whole-forward mean happened to be 2.10 ms lower than near. That reversal is evidence that approximately 1.08 seconds of whole-model execution noise dominates a one-call network effect at this sample size; it is not evidence that far placement is faster.

## How worker execution time is counted

The same expert's reported execution time differed across the nominally identical `c7i.xlarge` workers. For measured natural activations it averaged 1.048 ms near and 1.629 ms far. Worker compute is recorded separately and is not labeled network time. The average `unattributed` RPC component was 0.688 ms near and 1.258 ms far; host staging and restoration were both about 0.04 ms and 0.015 ms respectively.

The planner intentionally ranks observed end-to-end expert service time, because placement should respond to a slow host as well as a slow link. A topology-only claim would require CPU pinning and frequency/host controls, more repetitions, and a matched compute subtraction or independent network calibration.

## Method and environment

- Pinned `Qwen/Qwen3-30B-A3B` revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.
- Linux CPU BF16: PyTorch 2.2.2+cpu and Transformers 4.51.3.
- `r7i.4xlarge` coordinator with 128 GiB RAM; two identical `c7i.xlarge` expert workers.
- Prompt: `Reply with exactly one word: hello` (19 tokens).
- Original router, layer 0 expert 53, no forced routing.
- Five randomized local/near/far blocks with seed `20260820`.
- The full checkpoint came from an encrypted private S3 cache; only the coordinator synchronized all 16 shards, and workers downloaded the 9 MB expert artifact.
- Checkpoint load after synchronization took 16.11 seconds.

## Cleanup

Every node had a verified independent four-hour termination timer. The runner captured the result and destroyed all 13 run-scoped AWS resources. A separate inventory query found no live dAI instance or EBS volume, and the full-model OpenTofu state is empty. The input cache remains private and encrypted, with automatic expiration of every object after seven days.

## Claim boundary

This is real causal full-model execution with one naturally routed expert movable among three locations. It does not yet distribute all experts, test multiple hot experts, run decode/generation, measure concurrency or batching, or establish an end-to-end placement speedup. The next statistical gate is more blocks and more naturally selected experts; the next architectural gate is true multi-expert partitioning rather than module replacement for one expert.
