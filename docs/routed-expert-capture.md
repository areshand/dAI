# Routed-expert capture for placement analysis

The four-node low-VRAM runner separates performance measurement from routed-
expert instrumentation. The ordinary TP4/DP4/EP4 benchmark runs first without
capture. The runner then restarts the same topology with SGLang's
`--enable-return-routed-experts` flag and repeats the fixed cold-cache workload.
This prevents route collection overhead from being silently attributed to the
primary TPS result.

Run the existing disposable workflow:

```bash
./scripts/aws-low-vram-ep-experiment.sh
```

The instrumented phase writes:

- `routing-benchmark.json`: request metadata, output tokens, and SGLang's
  original base64 little-endian int32 route tensor with logical shape
  `[captured_token, layer, top_k]` for every warmup and measured request.
- `expert-routes.jsonl.gz`: one directly inspectable row per request, token
  position, and layer. Each row records the selected expert IDs and their
  current worker ranks.
- `expert-placement-summary.json`: the ten measured requests aggregated into
  per-layer expert activation counts, every observed same-layer expert-pair
  co-activation count, per-worker activation counts, and the distribution of
  how many workers each token-layer route spans.
- `routing-server-info.json` and `rank-*-routing-server.log`: proof that the
  capture server resolved `enable_return_routed_experts=true` and supporting
  per-rank diagnostics.

An expanded route row has this shape:

```json
{
  "request_index": 2,
  "server_request_id": "cmpl-example",
  "measured": true,
  "token_position": 1004,
  "phase": "decode",
  "layer": 17,
  "expert_ids": [3, 17, 40, 59, 72, 91, 108, 126],
  "worker_ranks": [0, 0, 1, 1, 2, 2, 3, 3]
}
```

For co-location, use same-layer pair counts rather than global expert IDs:
expert 17 in layer 4 and expert 17 in layer 5 are different weights. A pair
that is frequently selected together within one layer is a candidate for the
same low-latency region, subject to worker memory and load-balance constraints.
The `cross_worker_token_layer_fraction` and worker-fanout histogram quantify
how often the current contiguous EP4 placement requires multiple workers. The
runner explicitly selects SGLang's trivial placement and rejects results if
EPLB or redundant experts changed the logical-expert-to-worker mapping.

## Validity boundary

The default trace repeats one fixed 1,000-token benchmark prompt. It can show
co-activation for that workload and whether routing is stable across repeated
requests, but it cannot justify a general production placement. A claim-bearing
placement policy needs the same capture over a representative agentic-coding
prompt corpus, with separate training and held-out evaluation partitions. The
placement must be learned only from the training routes and evaluated on the
held-out routes and end-to-end inference results.
