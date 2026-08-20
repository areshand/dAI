# dAI MoE placement testbed

An experimental two-node testbed for measuring Mixture-of-Experts placement,
network latency, and correctness across heterogeneous Macs.

The core question is whether frequently selected experts should be placed near
the coordinator—and how a placement controller can use measured node latency,
expert popularity, compute time, and transfer cost to make that decision.

## What is here

- [`moe-distributed-experiment-design.md`](moe-distributed-experiment-design.md):
  the architecture, peer-discovery design, placement model, telemetry schema,
  experiment matrix, and staged implementation plan.
- [`prototype/`](prototype/): runnable synthetic and Qwen3 expert-boundary
  probes for a coordinator and one remote worker.
- [`prototype/EXPERIMENT-2026-08-20.md`](prototype/EXPERIMENT-2026-08-20.md):
  the first two-Mac correctness, latency, shadow-oracle, and paired-placement
  findings.

## Current evidence

The testbed has validated:

- deterministic output equivalence while swapping synthetic expert placement;
- exact BF16 CPU output equality for one extracted Qwen3 expert across two Macs;
- concurrent local-oracle and remote-shadow execution with matching generated
  tokens across three prompts;
- ten counterbalanced local-only/remote-only full-forward pairs; and
- Wi-Fi latency variance as a visible part of the placement cost, rather than a
  nuisance to hide.

The current two-node result is an experiment, not yet a general placement
policy. Network topology, worker load, expert batching, and model routing all
need broader coverage.

## Quick start: dependency-free transport probe

On the worker:

```bash
cd prototype
python3 two_node_moe.py serve --bind 0.0.0.0 --port 50123 --worker-id worker
```

On the coordinator:

```bash
cd prototype
python3 two_node_moe.py run \
  --remote-host worker.example \
  --port 50123 \
  --requests 400 \
  --payload-bytes 65536 \
  --concurrency 1 \
  --output results/two-node-baseline.json
```

Replace `worker.example` with a resolvable hostname or private address. The
service has no authentication or encryption and should only be exposed on a
trusted experiment network.

For the real-expert and full-model workflows, see
[`prototype/README.md`](prototype/README.md). They require a separately obtained
Qwen3 checkpoint and locally generated expert artifacts.

## Three-node AWS placement cell

The first cloud cell provisions a coordinator and near worker together in a
cluster placement group, plus a far worker in another Availability Zone. The
coordinator freezes application-path probes, derives a latency-aware placement,
and compares `hot_near`, `hot_far`, `latency_aware`, and seeded `random`
placements in counterbalanced blocks.

It defaults to the `mi:scratchpad` AWS profile and `us-west-2`:

```bash
./scripts/aws-status.sh
./scripts/aws-experiment.sh
```

The runner always invokes `tofu destroy` on exit and verifies that no live
run-tagged instance or EBS volume remains. Every instance also has a verified
systemd TTL timer and EC2's instance-initiated shutdown behavior is set to
`terminate`, providing cleanup if the local runner disappears. For recovery of
an interrupted local runner:

```bash
./scripts/aws-destroy.sh <exact-run-id>
```

The AWS cell is synthetic transport/placement evidence. It does not inherit
the full-model correctness claim from the Mac MPS experiment.

See the [first three-node AWS experiment](prototype/AWS-EXPERIMENT-2026-08-20.md)
for the measured discovery signal, counterbalanced policy comparison, cleanup
evidence, and claim boundary.

The real-model follow-up uses a memory-optimized Linux coordinator for the
pinned BF16 Qwen checkpoint and two identical CPU expert workers. It discovers
the real expert RPC path and counterbalances local, same-AZ, and cross-AZ
placements:

```bash
./scripts/aws-model-cache.sh --upload  # one-time protected seven-day cache
./scripts/aws-fullmodel-experiment.sh
```

The full checkpoint is synchronized from a private S3 input cache; only the
coordinator downloads all shards, while workers download the selected expert
artifact. The scratchpad cache is separate from ephemeral run output and has a
seven-day object-expiration policy.

See the [first real full-model AWS experiment](prototype/AWS-FULLMODEL-EXPERIMENT-2026-08-20.md)
for correctness evidence, decomposed expert timings, counterbalanced forward
results, and the remaining claim boundary.

## Repository hygiene

Model checkpoints, extracted weights, tensor outputs, virtual environments,
raw result files, and machine-local configuration are intentionally ignored.
The checked-in experiment report contains the shareable aggregate findings.

## License

MIT. Model checkpoints and other third-party artifacts retain their own
licenses and are not distributed in this repository.
