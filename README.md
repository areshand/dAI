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

## Repository hygiene

Model checkpoints, extracted weights, tensor outputs, virtual environments,
raw result files, and machine-local configuration are intentionally ignored.
The checked-in experiment report contains the shareable aggregate findings.

## License

MIT. Model checkpoints and other third-party artifacts retain their own
licenses and are not distributed in this repository.
