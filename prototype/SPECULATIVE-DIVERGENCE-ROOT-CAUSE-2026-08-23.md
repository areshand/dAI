# Temperature-zero speculative divergence: root-cause report

Date: 2026-08-23

## Finding

The output differs at temperature zero because temperature zero fixes the
**selection rule** (`argmax`); it does not require two GPU execution paths to
produce bit-identical logits.

For this Qwen3-30B-A3B/SGLang v0.5.16 configuration, ordinary generation and
speculative verification execute the same target model through different
Triton attention kernels:

- ordinary iterative generation uses `ForwardMode.DECODE` and
  `decode_attention_fwd`;
- n-gram speculation rewrites the batch to `ForwardMode.TARGET_VERIFY`, which
  the attention dispatcher routes through `forward_extend`; and
- with deterministic inference enabled, that extend path explicitly selects
  the unified one-stage extend kernel.

Those kernels are repeatable within their own paths, but they are not
numerically equivalent to each other in BF16. Their small logit differences
start on the first iterative token and eventually change the greedy argmax at a
near-tie. This is the causal root of the different output hash.

This is an implementation-level numerical parity issue in SGLang's target
verification path, not an inherent change introduced by the n-gram draft and
not probabilistic sampling.

## Decisive evidence

The controlled run was `dai-gen-20260823-012246` on one `g7e.4xlarge`, using
the pinned `lmsysorg/sglang:v0.5.16` image, Qwen3-30B-A3B BF16, TP=1, batch=1,
a fixed 1,000-token prompt, server seed 1234, PyTorch sampling, Triton attention,
and `--enable-deterministic-inference`.

The prompt hash and first output token were identical. The reported logprob of
the first output token was also identical at `-1.818568229675293`. Starting at
output position 1—the first step after prefill—the baseline and speculative
logprobs differed even while they continued emitting the same token. Examples:

| Output position | Emitted token | Baseline logprob | One-token NGRAM logprob | Delta |
|---:|---:|---:|---:|---:|
| 0 | 15442 | -1.818568 | -1.818568 | 0.000000 |
| 1 | 862 | -1.273063 | -1.247856 | +0.025207 |
| 2 | 5068 | -2.094006 | -1.949604 | +0.144402 |
| 7 | 4237 | -1.356348 | -1.481302 | -0.124954 |
| 13 | 13 | -0.000029 | -0.000027 | +0.000002 |

The first token-ID divergence occurred at output position 14, after an
identical prefix:

| Path | Token 34920 logprob | Token 20317 logprob | Greedy output |
|---|---:|---:|---:|
| Ordinary decode | -2.435763 | -2.435763 | 20317 |
| NGRAM target verify | -2.445220 | -2.570220 | 34920 |

The ordinary path reported an exact tie. Greedy `argmax` resolves the tie to
the lower vocabulary index, 20317. In target-verify mode, token 34920 had a
0.125 log-prob lead, so the same deterministic selection rule correctly chose
34920. Autoregression then amplified that one-token branch into a different
continuation.

The pinned SGLang source confirms the mechanism:

1. NGRAM changes the forward mode to
   [`TARGET_VERIFY`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/speculative/ngram_worker.py#L318-L328).
2. The attention dispatcher sends only `DECODE` to `forward_decode`; all other
   applicable modes, including `TARGET_VERIFY`, go to
   [`forward_extend`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/layers/attention/base_attn_backend.py#L178-L209).
3. Deterministic Triton extend selects the
   [unified one-stage kernel](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/layers/attention/triton_backend.py#L1292-L1304),
   while normal decode calls
   [`decode_attention_fwd`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/layers/attention/triton_backend.py#L1787-L1809).
4. Greedy verification uses the target logits' ordinary
   [`torch.argmax`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/speculative/eagle_utils.py#L648-L663).

## Controls and exclusions

All four NGRAM controls produced the same alternative output token stream and
first diverged from baseline at position 14:

| Variant | Draft tokens | Spec attention flag | CUDA graph | Same NGRAM hash |
|---|---:|---|:---:|:---:|
| `ngram-prefill` | 12 | prefill | yes | yes |
| `ngram-decode` | 12 | decode | yes | yes |
| `ngram-one-prefill` | 1 | prefill | yes | yes |
| `ngram-prefill-no-graph` | 12 | prefill | no | yes |

Therefore:

- it is not a relaxed acceptance threshold or a bad draft token: the target
  verifier's own logit for 34920 is highest;
- it is not draft-window length: one draft token per step still diverges;
- it is not CUDA-graph capture/replay: eager verify produces the same token
  stream; and
- it is not the speculative attention-mode flag: both settings still execute
  the `TARGET_VERIFY` forward geometry through the target's extend-attention
  path.

Prior runs also found the same position-14 split for NGRAM, uncompiled EAGLE3,
and standalone drafting. Because those methods use different sources for draft
tokens but share target verification, they independently support the same
conclusion.

## Meaning of deterministic inference

`--enable-deterministic-inference` makes a fixed runtime path reproducible. It
does not promise bitwise equality between different kernels, reduction orders,
batch geometries, or forward modes. Temperature zero then turns whatever
logits that path produced into one deterministic argmax. It cannot repair a
cross-path logit difference.

This distinction explains the observations:

- baseline is stable across repetitions;
- each speculative variant can also be stable across repetitions; yet
- baseline and speculative output hashes can differ.

## Consequence and next fix

The present speculative results cannot be described as exact-output speedups.
They may still preserve task-level quality, but that is a separate empirical
claim.

For exact greedy parity, target verification must compute logits through a
path numerically locked to ordinary decode. The smallest proof-of-fix is a
one-draft-token verifier that uses the same decode-attention kernel/metadata as
baseline and passes a token-by-token top-logit/argmax comparison. Only after
that cell passes should wider trees, EAGLE, remote drafting, or multi-machine
scaling be requalified.

## Artifacts and cleanup

Local ignored evidence is stored under:

`prototype/results/aws-generation/dai-gen-20260823-012246/`

The logprob reports, full token IDs, server configuration, server logs, and
comparison are preserved there. The runner destroyed all ten run-scoped AWS
resources and verified that no live EC2 instance or EBS volume remained.
