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
- [`docs/routed-expert-capture.md`](docs/routed-expert-capture.md): request-,
  token-, and layer-level Qwen MoE route capture plus co-location analysis for
  the four-node low-VRAM testbed.

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

`aws-status.sh` also reports the estimated current hourly burn, accrued EC2 and
live EBS cost, and the maximum exposure through each run's `ExpiresAt` TTL. Its
built-in rates cover the repository's default `us-west-2` instance types. Add a
custom rate without changing the script with, for example,
`DAI_EC2_HOURLY_RATES_JSON='{"custom.large":1.25}'`. The estimate excludes S3,
network transfer, taxes, credits, and account discounts.

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

## Scratchpad generation speed experiment

The earlier 1.08-second measurement is a 19-token, `use_cache=False` forward
pass. It is retained as a placement correctness control, but it cannot measure
time to first token (TTFT), autoregressive decode throughput, or speculative
decoding. The generation experiment uses one fixed, exactly 1,000-token prompt,
forces 256 output tokens, and reports TTFT, end-to-end latency, and output
tokens/second separately.

The disposable AWS runner compares four batch-one configurations on the same
GPU node and pinned SGLang container:

- ordinary SGLang tensor-parallel serving;
- n-gram speculative decoding, which needs no draft weights; and
- compiled EAGLE3 using a revision-pinned draft head trained for the exact base
  model; and
- standalone speculative decoding with Qwen3-0.6B as the draft model.

```bash
aws sso login --profile mi:scratchpad
./scripts/aws-generation-experiment.sh
```

The default `g5.12xlarge` has enough aggregate GPU memory for the 57 GB BF16
checkpoint. The runner refuses an instance whose current on-demand hourly price
exceeds `DAI_MAX_HOURLY_USD` (default `$8`), verifies the private S3 checkpoint
cache before provisioning, installs an independent instance TTL, destroys all
run resources on exit, and checks for surviving EC2 instances and EBS volumes.
Raw per-run artifacts remain ignored under `prototype/results/aws-generation/`.

The first complete qualification used one `g7e.4xlarge` in `us-east-2` because
the requested multi-GPU types were capacity-constrained. It found no valid 10×
speedup: n-gram reached 5.49× raw decode throughput but did not reproduce a
single exact output, compiled EAGLE3 reached 1.59× but diverged from the
baseline output, and the standalone draft was slower. See the
[generation speed report](prototype/AWS-GENERATION-EXPERIMENT-2026-08-22.md).

The follow-up research closed all three candidate directions against explicit
correctness, latency, and minimum-improvement gates. Its best raw result was
NGRAM-16 at 6.39× versus a deliberately deterministic Triton diagnostic
baseline. That 59.04 tok/s cell is not the production-speed baseline.

A fresh target-only qualification restored SGLang's optimized defaults and
measured **168.25 pooled output tok/s** on the same model and G7e. Every one of
ten cold-cache requests had exactly 1,000 uncached input tokens and 256 output
tokens; mean TTFT was 66.1 ms and mean end-to-end latency was 1.582 s. No draft,
speculation, quantization, or extra machine was involved. See the
[100 tok/s qualification](prototype/TARGET-ONLY-100-TPS-2026-08-24.md), the
[final execution report](prototype/RESEARCH-EXECUTION-2026-08-22.md) and
[closed research ledger](prototype/RESEARCH-TODO-2026-08-22.md).

The target-logprob follow-up isolated the temperature-zero divergence to
numerical differences between SGLang's ordinary Triton decode-attention path
and the Triton extend-attention path used by `TARGET_VERIFY`. Reproduce the
short control matrix with:

```bash
DAI_VARIANT_SET=root-cause ./scripts/aws-generation-experiment.sh
```

See the
[root-cause report](prototype/SPECULATIVE-DIVERGENCE-ROOT-CAUSE-2026-08-23.md)
for the token-level evidence and exclusions.

A valid 10× claim requires the same generation harness and metric on both sides.
It must not divide GPU decode tokens/second by the old one-forward latency.

### Quality-qualified 100 tok/s gate

The provider-oriented gate treats exact BF16 token identity as a debugging
signal, not as the product-quality definition. It runs the fixed speed contract
and a separate objective-answer suite against the baseline, NGRAM-16, and
compiled EAGLE3, then requires all of:

- at least 100 paired quality cases;
- a paired-bootstrap 95% lower bound no worse than two percentage points below
  baseline quality;
- at least 100 mean generated tokens/second;
- p99 visible stream-event gap below 100 ms; and
- p99 TTFT below 250 ms.

Run the included pipeline smoke suite with:

```bash
DAI_VARIANT_SET=quality ./scripts/aws-generation-experiment.sh
```

The bundled 16-case suite cannot qualify a result because it is intentionally
below the 100-case minimum. For a claim-bearing run, point the runner at a
representative JSONL suite:

```bash
DAI_VARIANT_SET=quality \
DAI_QUALITY_DATASET_LOCAL=/absolute/path/to/quality-suite.jsonl \
./scripts/aws-generation-experiment.sh
```

See [`prototype/quality/README.md`](prototype/quality/README.md) for the dataset
schema and claim boundary. Quality prompts retain their natural lengths; speed
remains measured on the fixed 1,000-input/256-output-token workload.

## Repository hygiene

Model checkpoints, extracted weights, tensor outputs, virtual environments,
raw result files, and machine-local configuration are intentionally ignored.
The checked-in experiment report contains the shareable aggregate findings.

## License

MIT. Model checkpoints and other third-party artifacts retain their own
licenses and are not distributed in this repository.
