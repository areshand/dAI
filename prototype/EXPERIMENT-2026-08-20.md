# Two-node synthetic MoE placement experiment — 2026-08-20

## Scope

This is an infrastructure result only. It validates two-node connectivity,
deterministic expert-output equivalence, placement swapping, binary payload
transport, and result capture. It does **not** measure Qwen inference, token
generation, or answer quality.

## Nodes

| Role | Host | Hardware | OS / Python |
|---|---|---|---|
| Coordinator | private-LAN coordinator | Apple M4 Max, 128 GiB unified memory | macOS 26.5.2 / Python 3.13.7 |
| Expert worker | private-LAN worker | Intel Core i7-4770HQ, 16 GiB RAM | macOS 12.7.6 / Python 3.9.6 |

## Fixed experiment

- Four deterministic synthetic experts.
- Experts 0 and 1 receive 327 of 400 requests in the sealed trace (81.75%).
- `hot_local` places experts 0 and 1 on the coordinator.
- `hot_remote` places experts 0 and 1 on `x`.
- Identical request IDs, expert routes, activation bytes, and expert digest
  functions are used for both placements.
- Every cell verifies that outputs are identical across placements.

## Isolated results

| Payload / concurrency | Placement | Remote requests | Overall p50 | Overall p95 | Requests/s |
|---|---|---:|---:|---:|---:|
| 4 KiB / 1 | hot local | 73 | 0.003 ms | 4.726 ms | 1,213 |
| 4 KiB / 1 | hot remote | 327 | 4.073 ms | 6.084 ms | 252 |
| 64 KiB / 1 | hot local | 73 | 0.034 ms | 5.900 ms | 851 |
| 64 KiB / 1 | hot remote | 327 | 5.792 ms | 7.667 ms | 195 |
| 64 KiB / 4 | hot local | 73 | 0.042 ms | 12.038 ms | 1,969 |
| 64 KiB / 4 | hot remote | 327 | 7.581 ms | 13.024 ms | 552 |

All three isolated cells passed output-equivalence validation. The 64 KiB,
concurrency-1 remote branch had a 5.85 ms median when the hot experts were
remote, compared with about 0.034 ms for local execution. Concurrency increased
aggregate throughput but also increased remote queueing/tail latency.

Two earlier supplemental files whose names lack `isolated` were accidentally
run concurrently against the same worker. They are preserved as contention
observations but excluded from the table and from placement conclusions.

## Interpretation

The result establishes that placement is a live, measurable variable in the
two-Mac harness and that the harness can hold synthetic outputs constant while
placement changes. It does not yet establish an optimal placement strategy.
The very large local/remote ratio is partly an artifact of a nearly zero-cost
synthetic local expert; a real expert's compute time will reduce that ratio.

## Real Qwen expert gate

The follow-up extracted `model.layers.0.mlp.experts.0` directly from the pinned
`Qwen/Qwen3-30B-A3B` revision
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`:

- Source shard: `model-00001-of-00016.safetensors`
- Source shard SHA-256:
  `454e77b346a61bfb201d54df60e15158838cf930617ee135113556204f2802b5`
- Expert artifact SHA-256:
  `446100403cb694b763af164c2b6c956a99d7ce5099dc34407d6ba3830260a692`
- Three BF16 matrices, 4,718,592 parameters, 9,437,776 artifact bytes
- Input fixture: 8 rows × 2,048 hidden elements, BF16, 32,768 bytes

Both machines used Python 3.9.6, PyTorch 2.2.2, Transformers 4.51.3,
NumPy 1.26.4, and Safetensors 0.5.3. Direct CPU execution produced exactly
equal output tensors (`max_abs=0`, `relative_l2=0`, cosine similarity 1.0):

| Execution | Mean expert compute |
|---|---:|
| M4 Max CPU | 6.37 ms |
| Intel i7 CPU on `x` | 9.47 ms |

The same real BF16 activation was then sent to a persistent TCP expert service
on `x` and its real expert output returned:

| Component | Mean |
|---|---:|
| End-to-end round trip | 15.34 ms |
| Worker expert compute | 10.60 ms |
| Non-worker transport/serialization/control | 4.73 ms |

The LAN-returned output was also exactly equal to the local reference tensor.
This validates one real expert boundary, not full-model distributed inference.

PyTorch 2.2.2 does not support BF16 on MPS, although BF16 works on both CPU
backends. Full-model acceleration therefore requires a newer Studio-only
PyTorch environment and a separately characterized cross-version/backend
tolerance; the pinned BF16 model must not be silently converted to FP16.

## Full-model MPS and one remote expert

A separate Studio-only environment used PyTorch 2.7.1 with the same
Transformers 4.51.3 model implementation. All sixteen pinned checkpoint shards
were downloaded and validated: 18,867 indexed tensor keys were present with no
missing keys. The full model loaded on MPS in BF16:

- MPS model allocation: 61,064,246,528 bytes
- Recommended MPS maximum: 115,448,725,504 bytes
- First load: 125.69 seconds
- Prompt: `Reply with exactly one word: hello` (19 tokens)
- First greedy token: ID 6023, text `hi`
- Initial one-token generation: 9.86 seconds

The extracted expert kernel on MPS averaged 0.65 ms for the eight-row fixture.
Against the PyTorch 2.2.2 CPU BF16 result, it passed the smoke tolerance with
`max_abs=0.00048828125`, relative L2 `5.04e-5`, and cosine similarity
`0.9999999987`.

An unmodified router capture selected layer 0 expert 53 for 7 of the 19 prompt
rows and ranked it first for the final prompt token. Expert 53 was therefore
extracted and deployed to `x`; routing was not forced. Its artifact SHA-256 is
`597e1ef02e6e40119bc46dfe982ce8ed4b91fa1970bbb68e9993d042fe1e69e2`.

The full model then ran the identical prompt twice in one process: once with
all local MPS experts, and once with only `(layer 0, expert 53)` replaced by the
BF16 CPU service on `x`. The final warmed run produced:

| Check | Result |
|---|---|
| Remote calls | 1 |
| Remote activation rows | 7 |
| Request / response bytes | 28,672 / 28,672 |
| Remote round trip | 19.89 ms |
| Worker compute | 12.48 ms |
| Worker preprocessing + serialization | 0.23 ms |
| Residual transport/client overhead | 7.18 ms |
| Selected routes across all token-layers | exactly equal |
| Final logits | bit-for-bit equal (`max_abs=0`) |
| Greedy token | exactly equal (`hi`) |

The server is warmed before admitting measurements. Earlier cold first-call
results of 125–350 ms are retained as cold-start observations and are not used
as steady-state network estimates. The observed 415 ms whole-forward difference
is also not a causal placement estimate: it came from one fixed-order local-then-
remote pair, while the directly timed RPC was only 19.89 ms. Counterbalanced
repetitions are required before interpreting end-to-end deltas.

## Concurrent shadow-oracle evaluation

The next validation ran the local MPS oracle and remote CPU expert concurrently
from the same staged activation. The model always consumed the oracle output;
the remote output was compared in shadow mode. Three prompts were evaluated,
with two shadow-generation repetitions each:

| Prompt | Oracle/shadow output |
|---|---|
| Reply with exactly one word: hello | `hi` |
| What is 2 + 2? | `4` |
| Capital of France | `Paris` |

Expert 53 was naturally selected once during prefill for every shadow
generation, producing six oracle/remote comparisons over 7–8 rows. Results:

- All six remote outputs passed BF16 tolerance.
- Two activation patterns were bit-exact.
- Maximum observed absolute difference was `0.00006103515625`.
- Maximum observed relative L2 was `3.19e-5`.
- All generated token sequences matched the oracle.
- After initial thread startup, local and remote branches began within about
  `0.03 ms` of one another.
- Local MPS expert completion was about `0.28–0.49 ms`; remote completion ranged
  from `17.27–126.36 ms` despite stable worker compute around `12–13 ms`.

This validates simultaneous shadow execution for correctness. It is not a
placement-performance cell because the duplicate oracle remains active.

## Balanced placement evaluation

Ten paired blocks then compared local-only and remote-only execution. Each block
contained both paths in randomized order, using the same loaded model, prompt,
router, expert, worker, and seed. Every measured run preserved exact selected
routes and token ID; all final logits passed tolerance.

| Metric | Local only | Remote only / delta |
|---|---:|---:|
| Mean full-forward time | 2,926.60 ms | 3,069.05 ms |
| Median full-forward time | 2,928.05 ms | 2,987.92 ms |
| Paired mean overhead | — | 142.44 ms |
| Paired median overhead | — | 74.14 ms |
| Paired overhead range | — | -25.49 to 401.06 ms |
| Paired-bootstrap 95% interval for mean | — | 56.33 to 237.65 ms |

The mean remote expert boundary was `105.74 ms`, with stable expert compute but
highly variable transport time. Fast RPC cells completed in `16–29 ms`; three
of ten calls took `216–298 ms`.

## Network-path diagnosis

The coordinator routes to the worker over its `en0` Ethernet interface. The
worker uses its `en0` Wi-Fi interface on an 80 MHz channel. Signal was strong (`-48 dBm`,
noise `-92 dBm`), but fifty ICMP samples showed:

| Samples | Loss | Minimum | Mean | Maximum | Std. dev. |
|---:|---:|---:|---:|---:|---:|
| 50 | 0% | 2.463 ms | 71.642 ms | 310.691 ms | 99.656 ms |

The ICMP tail matches the RPC tail, so the dominant variance is the Wi-Fi leg,
not expert computation or tensor serialization. This Wi-Fi setup is useful as a
named consumer-network topology cell, but it is not a stable LAN control. A
wired Ethernet or Thunderbolt link to `x` is required before estimating the
minimum local-placement overhead.

## Next gates

1. Add counterbalanced placement order and repeated runs so drift/order cannot
   masquerade as a placement effect.
2. Record request serialization, network round trip, worker queue, and worker
   compute separately.
3. Measure a payload-size/concurrency matrix and add controlled worker delay.
4. Alternate local and remote expert implementations within one loaded model,
   counterbalance order, and collect repeated paired forwards.
5. Time MPS→CPU staging, client serialization, socket transit, server stages,
   response transit, and CPU→MPS restoration independently.
6. Extend the validated boundary to multiple naturally selected experts and
   worker-level batching, then run a complete greedy generation loop with KV
   cache and per-token routing telemetry.
7. Repeat the sealed ten-pair placement cell after wiring `x`; keep the current
   Wi-Fi result as a separate topology rather than overwriting it.
