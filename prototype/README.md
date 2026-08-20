# Two-node synthetic MoE placement probe

This prototype tests the experiment infrastructure on a coordinator Mac and
an Intel Mac worker. It is intentionally dependency-free and compatible
with Python 3.9 and newer.

It holds a deterministic route trace and expert function constant, then compares:

- `hot_local`: the two experts receiving approximately 80% of requests run on
  the coordinator.
- `hot_remote`: those same hot experts run on the worker.

The output digest must be identical across placements. Any timing difference is
therefore attributable to the changed local/remote assignment within this
synthetic harness. This is a transport and placement probe, not evidence about
Qwen inference performance or answer quality.

## Run

On the worker:

```bash
python3 two_node_moe.py serve --bind 0.0.0.0 --port 50123 --worker-id worker
```

On the coordinator:

```bash
python3 two_node_moe.py run \
  --remote-host worker.example \
  --port 50123 \
  --requests 400 \
  --payload-bytes 65536 \
  --concurrency 1 \
  --output results/two-node-baseline.json
```

Increase concurrency only as a separately named experiment. The first baseline
uses concurrency 1 so queueing does not obscure network and placement effects.

## Real expert probe

`real_expert_probe.py` extracts one pinned Qwen3 expert, creates a deterministic
BF16 activation fixture, runs it locally or remotely, and compares output
tensors. See `EXPERIMENT-2026-08-20.md` for the exact artifact hashes and first
LAN result. Install the matching CPU comparison runtime with:

```bash
python3 -m venv .venv-real
.venv-real/bin/pip install -r prototype/requirements-real.txt
```

The source shard is retained locally under `model-cache/`; compact expert and
input fixtures are generated locally under `prototype/artifacts/`. Both are
excluded from Git because model weights and generated tensors are not source.

`full_model_smoke.py` loads the complete pinned checkpoint on MPS and can
capture the original router selections. `full_model_remote_expert.py` compares
an all-local full forward with a forward in which one naturally selected expert
runs on the worker. The coordinator-only MPS dependencies are pinned in
`requirements-studio.txt`; they intentionally live separately from the matched
PyTorch 2.2.2 CPU comparison environment.

`full_model_shadow_eval.py` runs the local expert oracle and remote expert
concurrently and returns the oracle output to the model. It is the correctness
mode. `full_model_placement_eval.py` disables duplicate execution and performs
balanced paired local-only/remote-only forwards. It is the performance mode.
