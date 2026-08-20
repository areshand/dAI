# Distributed MoE Inference Experiment Design

**Decision:** `DEC-MOE-DESIGN-001` revision 12  
**Plan:** `PLAN-REAL-E2E-MVP`  
**Design status:** proposed greenfield contract; no implementation exists yet  
**Revision 12 changes:** Phase 1 hardware assumption widened from a Linux/NVIDIA cluster to a heterogeneous local Mac and/or Linux cluster (CUDA, Apple Silicon/MPS, or CPU) with no NVIDIA dependency, so the team can iterate on hardware already on hand; gRPC is retained as the Phase 1 transport with libp2p/other P2P protocols explicitly reassigned to Phase 2 transport research; Phase 1 scope is widened to require a minimal two-placement latency/correctness comparison (not just single-placement correctness), and Phase 2A is reframed as a combined placement-optimization and transport/bottleneck research campaign.  
**Normative requirements:** captured from the original project brief.  
**Research-source access date:** 2026-08-19

## 1. Executive decision

The project will first build the smallest system that can answer one defensible scope question. This is not the owner-gated first primary hypothesis or its statistical acceptance contract:

> With a pinned Qwen3-30B-A3B model and its original router, can measured directional application-path latency plus correctness-qualified expert popularity and same-layer co-activation produce a feasible static placement that improves held-out, correct end-to-end behavior against preregistered placement policies while runtime replica selection, batching, topology, and workload remain frozen?

The selected MVP is one coordinator plus independent expert workers on a small heterogeneous cluster of locally available Mac and/or Linux machines — NVIDIA CUDA, Apple Silicon/MPS, or CPU-only workers — with no NVIDIA-specific dependency assumed for Phase 1; this lets the design iterate on hardware the team already controls rather than waiting on a Linux/NVIDIA procurement. The coordinator runs the complete causal shared Transformer path, all original routers, KV cache, routing-weight application, output combination, logits, sampling, and streaming. A replacement only at each sparse-MoE boundary dispatches real activation rows to workers holding exact `(layer_id, expert_id)` weights. Workers return raw, unweighted outputs. The data plane uses reused asynchronous unary `grpc.aio` calls over TCP, protobuf metadata, and contiguous binary tensor payloads; gRPC is chosen for Phase 1 because its maturity and simplicity suit correctness proof on a fixed, fully-known topology, while libp2p and other peer-to-peer network protocols are deferred to Phase 2 transport research, where they are evaluated for their relevance to the eventual decentralized dAI inference architecture. Static explicit placement and deterministic `FixedPrimary` replica selection are the Phase 1 controls. Directional traffic shaping is applied to the actual RPC path using Linux `tc`/`netem` where a leg runs on Linux; an equivalent macOS-native shaping mechanism (e.g., Network Link Conditioner, or `pf`/`dnctl` dummynet) is required wherever a leg runs on a Mac host, and is an explicit Phase 1 gate rather than an assumed default.

`LIVE_E2E` across physical hosts is mandatory before captured-activation execution, replay, or simulation. Only a validated `LIVE_E2E` run can establish distributed-model correctness. Network emulation can characterize the tested software under an impairment profile; it cannot establish physical geography.

Phase 1 has two goals, not one: prove that expert execution can be correctly distributed across independent machines, and prove that different static placements measurably change directional application-path latency and generated-output correctness/quality. Beyond the correctness MVP, Phase 1 therefore exercises at least two manual explicit placements — for example a naive/colocated placement against one deliberately different alternative — on identical hardware, topology, and workload, and reports the resulting latency and correctness deltas. This comparison is intentionally narrow: it establishes that placement is a live, measurable variable, not a statistically sealed placement-optimization result.

Phase 2 is the placement-optimization and networking research phase. The central research loop is: discover directional application-path latency on the actual data-plane; estimate prefill/decode-stratified hotness and co-activation from grouped training and preregistered validation traces; seal those inputs with model, hardware, topology, service calibration, and a held-out-test commitment; compute a complete feasible static plan offline; apply it as an immutable verified startup state; then compare policies on sealed held-out `LIVE_E2E` cells to converge on an optimal placement strategy. In parallel, Phase 2 evaluates libp2p and other peer-to-peer network/sync protocols as candidate replacements for gRPC in the eventual decentralized dAI architecture, and uses the same measured critical-path telemetry to identify the key latency and throughput bottlenecks in the distributed data plane. The planner never sees held-out test routes or outcomes, live queues or health, or policy-specific fresh measurements. Runtime replica selection remains the separately frozen scheduler factor.

MVP success means:

1. The pinned local reference and distributed run use identical model revision, precision, batch shape, prompts, generation loop, and correctness criteria.
2. Every real token-layer route is accounted for, and exact expert identities are preserved.
3. Generated greedy token IDs are exact and preregistered tensor/logit checks pass.
4. The actual RPC path is directionally shaped and verified by read-back, counters, and probes.
5. Every reported metric is reproducible from immutable resolved factors and raw typed events.
6. Placement, scheduler, topology, batching, and execution mode can each be varied alone and rejected by an isolation validator if another factor drifts.
7. A placement-study result binds one sealed input snapshot, a pure policy specification, a complete feasibility proof, an immutable startup-application record, and a comparison contract; only validated held-out `LIVE_E2E` observations can support real placement-performance claims.

This path deliberately excludes migration, Kubernetes, RDMA, distributed stores, and dashboards until a measured gap requires them. Consumer Mac hardware is included from Phase 1 as a first-class worker/coordinator host, not deferred; only large-scale consumer-fleet operation is deferred.

### Authority labels

- **Normative (N):** required by the local requirements or binding decision contract.
- **Verified (V):** supported by a pinned or primary external source.
- **Derived (D):** arithmetic from verified inputs, not a measurement.
- **Proposed (P):** selected design contract to be implemented and tested.
- **Gated (G):** unresolved project-owner or feasibility decision.

## 2. Proof rules and claim enforcement

Every run has one primary `execution_mode` and the complete fidelity vector below. A result analyzer loads both the mode and validation record, then evaluates a versioned claim allowlist. A report build fails if it contains a claim class not allowed by that pair.

| Mode | Required fidelity | Allowed claims after its own validation | Prohibited claims |
|---|---|---|---|
| `LIVE_E2E` | `FULL`, `ORIGINAL_LIVE`, `LIVE_CAUSAL`, `REAL`, physical or emulated network, `WALL_CLOCK_MEASURED` | Distributed correctness against the named baseline; complete latency/throughput for the tested system; scheduler and placement behavior | Untested hardware/topologies; physical geography when network is emulated |
| `CAPTURED_ACTIVATION_EXECUTION` | `EXPERT_ONLY`, `CAPTURED`, `CAPTURED`, `REAL`, physical or emulated network, `WALL_CLOCK_MEASURED` | Expert-output tolerance, RPC/worker cost for captured tensors, batching and replica behavior | Router fidelity, causal generation, generated quality, full-model latency or correctness |
| `FORCED_ROUTE_EXECUTION` | `FULL` or `EXPERT_ONLY`, `FORCED`, live or captured activations, declared compute/network/timing | Instrumented forced-routing behavior; quality only after a separate quality evaluation | Unmodified-Qwen correctness or original-router fidelity |
| `EVENT_REPLAY` | `NONE`, `CAPTURED`, `NONE`, measured profile or modeled compute, physical/emulated/simulated network, `TRACE_DRIVEN` | Deterministic comparisons within the replay inputs and calibration | Tensor correctness, generated output, absolute hardware performance, real inference |
| `DISCRETE_EVENT_SIMULATION` | `NONE`, synthetic or captured routing, no live activations, `MODELED`, `SIMULATED`, `MODELED` | Comparative hypotheses within the stated model | Production RPC behavior, real computation, real latency, model quality |
| Orthogonal `network=EMULATED` label | Any compatible primary mode whose actual data-plane RPC crosses validated shaping | Software behavior under the applied, probed impairment profile | Physical geography, Internet routing, or unmodeled WAN behavior |

The fidelity vector is mechanically constrained:

```text
model_path:  FULL | EXPERT_ONLY | NONE
routing:     ORIGINAL_LIVE | CAPTURED | FORCED | SYNTHETIC
activations: LIVE_CAUSAL | CAPTURED | SYNTHETIC | NONE
compute:     REAL | MEASURED_PROFILE | MODELED
network:     PHYSICAL | EMULATED | SIMULATED
timing:      WALL_CLOCK_MEASURED | TRACE_DRIVEN | MODELED
```

Claim rules are not advisory:

- `distributed_model_correctness` requires `execution_mode=LIVE_E2E`, the exact fidelity tuple above, and analyzer predicates `baseline_equivalent`, `token_ids_exact`, and `numerical_tolerances_pass` in the verified `validation-record.v1`.
- `generated_quality` requires a causal full model path; if routing is not `ORIGINAL_LIVE`, it also requires an analyzer-derived quality-comparison predicate against unmodified routing. No such evidence may be emitted until the conditional `quality-evaluation-record.v1` contract is activated in a new closed registry; it remains gated here.
- `physical_geography_performance` requires `network=PHYSICAL`, recorded physical sites, and no emulation-only evidence substitution.
- `placement_policy_performance` requires a held-out validated `LIVE_E2E` cell, a verified `placement-comparison-contract.v1`, a sealed test commitment predating every policy/plan/comparator hash, exact frozen-factor equality, and post-run token/route equality. Replay may screen candidates but cannot enable this claim.
- `one_way_latency` requires `clock_uncertainty_ns <= preregistered_max_clock_uncertainty_ns`; otherwise the field is suppressed.
- Missing claim-required events, telemetry drops, unresolved routes, or unmatched required spans cause the analyzer to derive `run_validity=INVALID` and prevent metric publication.

### 2.1 Analyzer-produced validation evidence

`validation-record.v1` is a content-addressed output of `ResultAnalyzer`, never a user-authored input. Its existing rule `validation_record_id = sha256(JCS(record_without_validation_record_id))` is exactly the §3 global projection because the registry names `validation_record_id` as its sole `own_identity_field`; it is not a parallel convention. The immutable run manifest, raw-event manifest, baseline comparison, topology proof, analyzer binary/container, schema bundle, claim rules, and completeness-rule hashes remain in that projection. The record contains:

```text
schema=validation-record.v1, validation_record_id, produced_at_utc
analyzer={name, version, source_revision, binary_sha256,
          container_digest={uri, digest, expected_identity_kind=native_digest,
                            expected_type_or_declaration=RAW-07_OCI_IMAGES,
                            [retained_bytes={uri, digest, expected_identity_kind=raw_sha256,
                                             expected_type_or_declaration=RAW-07_OCI_IMAGES}]}}
run_manifest={schema=run-manifest.v1, run_id, uri, artifact_id}
schema_bundle={schema=schema-bundle.v1, uri, artifact_id}
claim_rules={schema=claim-rules.v1, uri, artifact_id}
completeness_rules={schema=validation-completeness-rules.v1, uri, artifact_id}
raw_event_manifest={schema=raw-event-manifest.v1, uri, artifact_id}
evidence[]={kind, uri, sha256, expected_identity_kind,
            expected_type_or_declaration, schema, byte_count}
event_completeness[]={predicate_id, required_event_types, expected_count,
                      observed_count, unmatched_ids, telemetry_drop_count, outcome}
topology_proof={topology_proof_id, uri, sha256,
                expected_identity_kind=structured_jcs,
                expected_type_or_declaration=topology-proof.v1, outcome}
topology_probe_results[]={probe_id, ordered_link_id, evidence_sha256,
                          observed, expected_tolerance_rule_id, outcome}
baseline_comparison={comparison_id, uri, sha256,
                     expected_identity_kind=structured_jcs,
                     expected_type_or_declaration=baseline-comparison.v1, outcome} | null
predicate_results[]={predicate_id, implementation_version,
                      input_hashes[]={uri, sha256, expected_identity_kind,
                                      expected_type_or_declaration},
                      observed, expected, outcome=PASS|FAIL|NOT_APPLICABLE}
claim_results[]={claim_class, required_predicate_ids[], outcome=ALLOW|DENY,
                  denial_reasons[]}
run_validity=VALID|INVALID, invalidity_reasons[]
```

`topology-proof.v1` is likewise analyzer-produced and content-addressed:

```text
schema=topology-proof.v1, topology_proof_id, produced_at_utc
run_manifest_sha256, topology_spec_sha256, topology_epoch
applied_rules[]={ordered_link_id, source_node_id, destination_node_id,
                  interface_id, filter/address_binding, tc_htb_ifb_netem_readback,
                  readback_sha256, physical_link_evidence_id|null}
counters[]={ordered_link_id, counter_kind, pre, post, evidence_sha256}
probe_results[]={probe_id, ordered_link_id, source, destination,
                  observed_delay, observed_bandwidth, observed_jitter, observed_loss,
                  tolerance_artifact_sha256, outcome}
claim_link_results[]={claim_class, required_ordered_link_ids[], outcome}
outcome=PASS|FAIL, failure_reasons[]
```

Its existing rule `topology_proof_id = sha256(JCS(record_without_topology_proof_id))` is likewise an exact alias of the §3 projection and registry entry. A topology claim requires all claim-specific rule bindings, counters, links, and probes to pass; logical-region labels are never topology proof. Physical-WAN evidence additionally binds a project-approved physical-site/link inventory artifact rather than inferring geography from delay or labels.

The claim evaluator is deny-by-default:

| Claim class | Required analyzer-derived predicates in addition to mode/fidelity | Default |
|---|---|---|
| `distributed_model_correctness` | `run_event_complete`, `all_routes_terminal`, `baseline_equivalent`, `token_ids_exact`, `numerical_tolerances_pass` | deny |
| `complete_latency_throughput` | `run_event_complete`, `critical_spans_complete`, `endpoint_counts_reconcile`, `run_validity_valid` | deny |
| `emulated_impairment_performance` | preceding metric predicates plus `topology_rules_bound`, `directional_probes_pass` | deny |
| `physical_geography_performance` | preceding metric predicates plus `physical_site_links_evidenced`, `no_emulation_substitution` | deny |
| `deterministic_replay_comparison` | `replay_inputs_complete`, `dependency_edges_complete`, `priority_table_bound`, `repeat_replay_byte_identical` | deny |
| `one_way_latency` | `clock_bound_pass` and the relevant complete local/cross-clock span predicates | deny |
| `placement_policy_performance` | `heldout_live_e2e`, `placement_inputs_sealed`, `policy_access_conformant`, `no_online_placement_updates`, `placement_feasible`, `startup_application_verified`, `comparison_factors_equal`, `live_route_trace_equal` | deny |

Unknown claim classes, predicates with missing evidence, `NOT_APPLICABLE` required predicates, hash mismatches, analyzer failures, or incomplete records yield `DENY`. Literal strings such as `PASS`, `VALID`, or “topology proof present” in a manifest, status file, or report have no authority. Only a successfully verified, content-addressed analyzer record can enable a claim, and report generation recomputes its ID and evidence hashes before use.

No expert-only kernel, mocked worker, synthetic activation, captured route, replay, forced route, or simulation result may be labeled “real end-to-end inference.”

## 3. Research validity and independent factors

The overarching invariant is that model execution, physical expert placement, network topology, scheduling, and batching remain separate abstractions. The complete resolved factor bundle is content-addressed using canonical JSON ([RFC 8785](https://www.rfc-editor.org/rfc/rfc8785)); run attempts use UUIDv7 ([RFC 9562](https://www.rfc-editor.org/rfc/rfc9562.html)).

For every registered structured artifact type `T`, the one global identity rule is:

```text
f = artifact-identity-registry[T].own_identity_field  # one exact root field, or none
hash_projection(o,T) = o with exactly root field f removed, if f is not none;
                       otherwise o unchanged
artifact_id(o,T) = sha256(JCS(hash_projection(o,T)))
```

The registry in §11 is part of the hashed schema bundle and has exactly one row per artifact type: `own_identity_field` is either one exact root-field name or literal `none`. No wildcard, alternate name, nested alias, or second registration is permitted. If a field is registered, a resolved artifact must contain it exactly once and its value must equal the lowercase 64-hex recomputation; the implementation removes the field, never blanks or sentinel-fills it. If `none` is registered, the content identity is stored out of band in the immutable artifact index and parent references, and the artifact must not embed an own content-identity alias.

Only the registered root field is excluded. Semantic IDs such as `run_id`, `replica_id`, and `candidate_slot_id`, and every referenced child `*_id` or `*_sha256`, remain in the projection. Thus changing a child artifact ID changes its parent ID. Duplicate JSON keys, an absent or wrong registered own field, duplicate registry rows, an unregistered own-identity alias, self-inclusive hashing, `null`/blank/all-zero/`TBD`/`GATED_*` identity sentinels, or any placeholder convention is invalid. A schema-declared optional child reference may be absent or `null`, which means no edge and no indexed child; it is never accepted as a present identity. Serialization member order does not affect JCS or the resulting ID. Raw binary/Parquet/event and undeclared opaque evidence files remain byte-hashed out of band and cannot carry an embedded own identity; any JSON/YAML object used as a resolved config, factor, parent, equality input, or claim record must declare a registered type.

| Factor | Identity | Owns |
|---|---|---|
| `ModelSpec` | `model_id` | checkpoint/runtime/precision/router semantics |
| `WorkloadSpec` | `workload_id` | prompt manifest, arrival process, generation inputs |
| `HardwareInventory` | `hardware_id` | declared capacity and separately measured profiles |
| `TopologySpec` | `topology_id` | nodes, physical/logical location, ordered directional links |
| `PlacementPlan` | `placement_id` | desired expert replicas only |
| `SchedulerSpec` | `scheduler_id` | runtime replica choice only |
| `BatchingSpec` | `batching_id` | cohort, dispatch packing, worker microbatch stages |
| `ExecutionModeSpec` | `mode_id` | proof taxonomy and fidelity vector |
| `FaultPlan` | `fault_plan_id` | scheduled failure/churn controls |
| `CalibrationSet` | `calibration_id` | frozen measured service curves and link probes |
| `PlacementStudySnapshot` | `study_snapshot_id` | sealed planning inputs, common feasible universe, feature-access stages, and test commitment |
| `PlacementPolicySpec` | `placement_policy_spec_id` | offline feature allowlist, objective/constraints, solver identity and budget |

The `Identity` names in this factor table are logical aliases for the recomputed `artifact_id` carried by the run manifest/parent reference; they are not additional embedded own fields unless §11 explicitly registers one.

Every comparison declares exactly one `vary_only` from `placement_policy`, `scheduler`, `topology`, `batching`, or `execution_mode`. For `placement_policy`, the only permitted differences are the policy specification, its derived complete feasible plan, the plan-derived directory/placement epoch, and resulting observations. The isolation validator compares all other resolved factor hashes, including the sealed `constraint-set.v1` hash and `scheduler-spec.v1` hash, runtime/container fingerprints, seed schedules, discovery and calibration snapshots, test partition, readiness/reset state, exact replica/byte budget, and fault plan. It refuses aggregation if anything else differs. Observed queues, health, attempts, completion times, and runtime decisions are outcomes, never placement-policy inputs.

## 4. Pinned Qwen3-30B-A3B reference

### 4.1 Reproducibility tuple

The proposed reference is `Qwen/Qwen3-30B-A3B` revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, unquantized BF16, with `transformers==4.51.3`. The initial oracle should pin eager attention, `eval()`, `torch.no_grad()`, `use_cache=True`, and greedy decoding. Tokenizer, chat template, generation configuration, PyTorch, accelerator backend (CUDA, Apple MPS, or CPU) and its version, driver (where applicable), accelerator model, and container digest are also resolved run fields, not ambient state.

Primary pinned sources are the [model configuration](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/config.json), [model card](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/README.md), [safetensors index](https://huggingface.co/Qwen/Qwen3-30B-A3B/resolve/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/model.safetensors.index.json), [Transformers 4.51.3 implementation](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py), and [configuration class](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3_moe/configuration_qwen3_moe.py).

### 4.2 Verified facts

| Property | Verified value |
|---|---:|
| Total parameters | 30,532,122,624 |
| Decoder/MoE layers | 48 / 48 |
| Hidden size | 2,048 |
| Routed-expert intermediate size | 768 |
| Routed experts per layer | 128 |
| Selected experts per real token-layer | 8 |
| Attention heads / KV heads / head dimension | 32 / 4 / 128 |
| Vocabulary | 151,936 |
| Router behavior | FP32 softmax; `topk(k=8)`; `norm_topk_prob=true`; cast weights to hidden dtype |
| Expert operation | `down_proj(silu(gate_proj(x)) * up_proj(x))` |
| Shared routed expert | none in this checkpoint |
| Capacity dropping | none in the pinned inference implementation |
| Checkpoint dtype/format | BF16, 16 safetensors shards plus index |

“All 48 layers are MoE” follows from `decoder_sparse_step=1`, empty `mlp_only_layers`, and the pinned layer-selection condition. Transformers exposes router logits when requested, but it does not expose a stable remote expert-dispatch API, arbitrary partial expert processes, a replica scheduler, or an OpenAI server. The MVP therefore replaces the module boundary without editing upstream generated source. Named tensors can be read from safetensors through its [official loading API](https://huggingface.co/docs/safetensors/main/index), but exact partial-loader correctness remains a Phase 0 gate.

For `T=B*S`, the pinned sparse block receives `[B,S,2048]`, forms router logits `[T,128]`, selects IDs and normalized weights `[T,8]`, executes exact experts, applies each weight, and combines with `index_add_` into `[T,2048]` before reshaping. Each expert has three BF16 weights:

```text
gate_proj.weight [768,2048]
up_proj.weight   [768,2048]
down_proj.weight [2048,768]
```

### 4.3 Derived, not measured, accounting

| Quantity | Derivation | Result |
|---|---|---:|
| Expert tensors | `48 * 128 * 3` | 18,432 tensors |
| Parameters/expert | `3 * 2048 * 768` | 4,718,592 |
| BF16 bytes/expert | `4,718,592 * 2` | 9 MiB |
| Routed-expert parameters | `48 * 128 * 4,718,592` | 28,991,029,248 |
| Routed-expert BF16 storage | previous row `* 2` | 54.00 GiB |
| Centralized non-expert parameters | total minus routed experts | 1,541,093,376 |
| Centralized non-expert BF16 storage | previous row `* 2` | 2.87 GiB |
| Whole BF16 weights | total parameters `* 2` | 56.87 GiB |
| KV bytes/token/request | `48 * 2(K,V) * 4 * 128 * 2` | 98,304 bytes = 96 KiB, before overhead |
| Worst-case v1 route traffic/token | `48 * 8 * 2048 * 2 bytes` each direction | about 1.5 MiB each way, 3 MiB aggregate |

The 3.3B active-parameter figure is not a residency requirement: all 30.5B parameters must remain available unless a run explicitly uses cold/offloaded experts. “Shared weights” in this document means embeddings, attention, normalization, routers, output head, and other non-expert parameters—not a Qwen shared routed-expert branch.

## 5. Architecture and ownership

```text
OpenAI-compatible API adapter
          |
admission + fixed request cohorts + streaming/cancellation
          |
ModelRuntime: embeddings, attention, original routers, KV, LM head
          |
RemoteMoeBlock -> ExpertDirectory -> SchedulingPolicy -> DispatchBatcher
                                                        |
                                reused unary grpc.aio/TCP requests
                                                        |
                                    independent ExpertWorkers

PlacementPolicy -> desired residency -> workers -> ExpertDirectory
TopologyController -> real RPC-path shaping/failures/probes
Telemetry/ResultStore <- immutable factors, typed events, artifacts
ResultAnalyzer -> validation, metrics, statistics, plots, claim allowlist
```

| Component | Inputs and outputs | Owns | Forbidden ownership |
|---|---|---|---|
| `OpenAIAPIAdapter` | versioned HTTP request + connection lifecycle → canonical inference command; canonical output/error events → HTTP/SSE response | boundary parsing/validation, compatibility-manifest selection, request identity, disconnect/cancellation translation, ordered SSE framing | tokenization, generation defaults, cohorts, model execution, routing, placement, scheduling, worker retries |
| `ModelRuntime` | tokens/cache/context → logits/cache/events | full shared path, original routers, causal generation | placement, topology, worker queue policy |
| `RemoteMoeBlock` | `[B,S,2048]` → `[B,S,2048]`, `[B*S,128]` | exact router semantics, route construction, reordering, weighted combination | residency, replica scoring |
| `PlacementPolicy` | sealed placement-study snapshot + policy spec → desired plan or structured infeasibility | offline exact replica/tier assignments, objective components, feasibility proof | held-out test routes/outcomes, live queues/health/attempts, primary/fallback order, runtime choice, scheduling |
| `ExpertDirectory` | worker state events → immutable snapshot; lookup → candidates | observed residency, health, incarnation, revisions | desired placement or candidate scoring |
| `SchedulingPolicy` | sealed logical batches + decision context → replica/fallbacks/scores | choice among eligible exact replicas | changing expert IDs, loading weights, batching |
| `BatchingPolicy` | compatible work → cohorts, packed dispatches, kernel microbatches | three explicit versioned batching stages | placement, replica choice, topology |
| `TopologyController` | topology/fault plan → applied epoch and probes | link shaping, failure actions, state read-back | scheduling or placement |
| `Dispatcher/ExpertTransport` | packed request + deadline → response/failure | framing, channels, attempt execution, cancellation, duplicate suppression | retry policy or replica scoring |
| `ExpertWorker` | request → raw outputs + local spans | exact resident weights, bounded queues, expert kernels | routers, routing weights, combination, sampling |
| `Telemetry/ResultStore` | events/artifacts → immutable attempt | evidence integrity and content hashes | changing runtime decisions |
| `ResultAnalyzer` | complete immutable attempt → validation/metrics/plots | derivation, uncertainty, claim enforcement | mutation of raw evidence |

The coordinator loads every checkpoint key except `.mlp.experts.`. Workers load only explicitly assigned expert triples and verify the pinned model revision and tensor digests. `RemoteQwen3MoeBlock.forward` preserves the observable contract:

```python
forward(hidden_states: Tensor[B, S, 2048])
    -> tuple[Tensor[B, S, 2048], Tensor[B*S, 128]]
```

It copies the pinned FP32 softmax, top-8 selection, renormalization, dtype cast, reference row ordering, coordinator-side weighting, and accumulation semantics. Stock `from_pretrained()` materialization followed by deleting experts is permitted only as a temporary measured bring-up shortcut on hardware that fits it; the target is a manifest-driven partial loader.

## 6. Identities, snapshots, lifecycle, and errors

### 6.1 Identity hierarchy

```text
experiment_id -> trial_id -> run_id
trace_id -> span_id -> parent_span_id
session_id -> request_id -> token_phase + token_index/absolute_token_position
layer_id -> route_id + topk_slot + expert_id
logical_expert_batch_id -> worker_layer_dispatch_id -> attempt_id
worker_batch_id -> replica_id + worker_id + node_id
placement_epoch + topology_epoch + directory_revision + worker_incarnation
```

`traceparent` follows [W3C Trace Context](https://www.w3.org/TR/trace-context/) across API and RPC boundaries. Domain IDs remain explicit because trace/span IDs do not encode model semantics.

A scheduling decision receives one immutable `DecisionContext` containing `decision_id`, `logical_expert_batch_id`, decision monotonic time, directory/topology/placement revisions, candidate health, queue observations with source sequence and age, service-model revision, and policy seed. Directory and health entries come from one committed revision. Stale runtime observations are rejected or explicitly penalized according to `max_snapshot_age_us`; replay uses the captured decision context. That scheduler field never refreshes or invalidates the immutable directory within a Phase 1/2 controlled run. Placement-study discovery/calibration age is checked separately at experiment admission; a stale snapshot starts a new study version rather than a live refresh.

### 6.2 Placement and residency

Desired `PlacementPlan` and observed `DirectorySnapshot` are separate. Phase 1 and Phase 2 apply each static plan only at process startup through:

```text
RESOLVED -> PREPARED -> LOADING -> VERIFIED -> COMMITTED -> ADMISSION_OPEN
```

`PREPARED` re-hashes every input, proves complete expert coverage and per-node capacity/slot/compatibility constraints, and binds worker identities. `LOADING` occurs with admission closed. `VERIFIED` requires every assignment's model/expert digest, exact bytes, tier, residency generation, usable capacity, and worker incarnation to match and be `DEVICE_WARM`. `COMMITTED` atomically publishes one complete directory revision bound to the placement epoch; only then may admission open. A pre-commit failure unloads only replicas loaded for that attempt, publishes no directory revision, and leaves admission closed. Every request pins that immutable committed revision. A worker-incarnation change invalidates the run; there is no in-run repair.

Dynamic placement is documentation-only until an owner-approved Phase 4-or-later study. Its proposed contract is `PREPARE -> LOAD_ADDITIONS -> VERIFY_COMPLETE -> READY -> CAS_COMMIT -> DRAIN_REMOVALS -> UNLOAD -> FINALIZE`: retain old serving replicas through verification; prove peak transition headroom, slots, compatibility, and declared failure-set coverage; compare-and-swap the expected monotonic directory/placement epoch; pin each in-flight cohort to one committed snapshot; reject stale-epoch selections/responses; and drain removals only after no pinned work remains or an explicit deadline expires. Before commit, rollback aborts additions and leaves the old epoch. After commit while old replicas remain, rollback is a new monotonic forward epoch restoring them. After unload, recovery is a new prepare/load/verify/commit operation, never an epoch decrement. No migration daemon, controller service, or executed rollout claim is part of the MVP.

Residency states are:

```text
ABSENT -> STORED -> LOADING_HOST -> HOST_WARM
       -> LOADING_DEVICE -> DEVICE_WARM -> DRAINING -> ABSENT
                                           \-> FAILED
```

The worker is authoritative for observed state. Only `DEVICE_WARM` is eligible under the strict Phase 1 warm-state policy. A cold expert is never silently advertised as ready.

### 6.3 Eligibility and failures

A replica is eligible only when model and expert digests match, its placement epoch is committed, state is allowed, health is `SERVING`, incarnation is current, dtype/runtime are compatible, capacity admits the batch, and snapshot age is acceptable.

| Failure | Terminal contract and observable outcome |
|---|---|
| `PROTOCOL_MISMATCH` | fail attempt; do not retry incompatible endpoint |
| `MODEL_OR_DIGEST_MISMATCH` | hard fail; quarantine registration |
| `STALE_RESIDENCY_GENERATION` | refresh snapshot and explicitly replan within deadline |
| `EXPERT_NOT_RESIDENT` | replan only to an exact replica |
| `EXPERT_NOT_READY` | bounded wait or exact alternate replica |
| `RESOURCE_EXHAUSTED` | atomic rejection with `QUEUE_FULL`, `PINNED_BUFFER_FULL`, or `DEVICE_MEMORY_PRESSURE`; backpressure/replan |
| `DEADLINE_EXCEEDED` / `UNAVAILABLE` | new explicit attempt only if the request budget permits |
| malformed dtype/shape/order/cardinality | hard fail; never reinterpret bytes |
| duplicate/late response | discard by logical-dispatch and accepted-attempt state; charge bytes/compute |
| no exact replica | fail model request in correctness mode |
| cancellation | stop admission/new attempts; propagate cancellation; late work remains observable |

No fallback substitutes another expert, drops a route, uses a stale output, or renormalizes over fewer experts. A physical packed request is atomic in v1. On failure, its logical expert batches are replanned and repacked; every new physical attempt gets a new `attempt_id`, while logical IDs remain stable. Only one accepted result can contribute to combination.

## 7. Canonical component and RPC interfaces

```text
OpenAIAPIAdapter.accept(http_request, connection_lifecycle, compatibility_manifest)
  -> CanonicalInferenceCommand | CanonicalAPIError

OpenAIAPIAdapter.emit(canonical_output_events, connection_lifecycle)
  -> HTTPResponse | OrderedSSEStream | CanonicalAPIError

ModelRuntime.forward_step(inputs, cache, context)
  -> logits, updated_cache, runtime_events

ExpertDirectory.lookup(model_revision, layer_id, expert_id, directory_revision)
  -> immutable candidates

SchedulingPolicy.choose(logical_expert_batch, decision_context)
  -> selected_replica, ordered_fallbacks, candidate_scores, rejection_reasons

DispatchBatcher.pack(replica_decisions, batching_spec)
  -> ExpertLayerRequest[]

ExpertTransport.execute(request, deadline)
  -> ExpertLayerResponse | StructuredFailure

ExpertWorker.execute(request)
  -> unweighted outputs, worker spans, terminal status

PlacementPolicy.plan(placement_study_snapshot, placement_policy_spec)
  -> PlacementPlanV2 + PlacementFeasibilityProof | StructuredInfeasibility
```

`CanonicalInferenceCommand` contains only validated boundary data: `api_request_id`, selected compatibility version, model alias, ordered messages or prompt, resolved explicitly supplied generation fields, stream flag, client deadline, cancellation handle, and trace context. Output events are ordered deltas, terminal choice/reason, usage, or a structured internal error. The adapter owns lifecycle transitions `RECEIVED -> VALIDATED -> ACTIVE -> TERMINAL` and translates disconnect/cancel exactly once. It must not tokenize, inject unspecified defaults, schedule cohorts, own KV cache, call experts, retry worker RPCs, or reinterpret model errors. Exact accepted fields, defaulting rules, HTTP/SSE encoding, cancellation timing, concurrency, status codes, and public error compatibility remain owner gate 4; this canonical interface fixes ownership, not public behavior.

`PlacementPolicy.plan` is a deterministic offline pure function. Its snapshot exposes only preregistered training features and, where the policy spec permits, validation features used for hyperparameters, stopping, or candidate selection. The test partition is represented only by a sealed commitment until all candidate plans and comparator identities are hashed. Neither input nor output contains runtime primary/fallback ordering, mutable directory health, queue observations, dispatch/attempt decisions, or scheduling state.

### 7.1 Transport selection

The default is reused asynchronous unary `grpc.aio` over TCP. gRPC recommends channel reuse and its Python guidance notes that streaming may add thread overhead; each measured call has an explicit deadline and bounded message size ([gRPC performance guide](https://grpc.io/docs/guides/performance/), [deadlines](https://grpc.io/docs/guides/deadlines/)). Automatic retry, hedging, compression, and `wait_for_ready` are disabled. Tensor bytes remain on an ordinary kernel TCP path that Linux traffic control can shape.

Protobuf provides the versioned envelope; `bytes` holds arbitrary data and repeated-field order is preserved ([proto3 guide](https://protobuf.dev/programming-guides/proto3/)). Contiguous BF16 tensors use little-endian row-major bytes. Payload length must equal `row_count * hidden_dimension * dtype_bytes`. On CUDA workers, reusable page-locked host buffers enable asynchronous device copies as documented in the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf); on Apple Silicon/MPS workers the equivalent is unified-memory-backed buffers with `torch.mps` stream synchronization, and on CPU-only workers no host/device copy exists at all. This is host-staged transport, not zero-copy or GPU-direct on CUDA hosts.

gRPC over TCP is the selected Phase 1 transport because its maturity, code generation, and deadline/flow-control semantics are sufficient to prove distributed correctness on a small, fully-known, static topology without conflating transport-protocol validation with the correctness proof itself. libp2p and other peer-to-peer transport/discovery protocols are explicitly out of scope for Phase 1 and are the subject of Phase 2 transport research: they matter for the eventual decentralized dAI inference architecture, where peer discovery, NAT traversal, and multiplexed streams over untrusted/dynamic topologies replace the fixed, operator-known Phase 1 deployment. No transport substitution inherits the Phase 1 gRPC evidence label automatically.

The example's 32 MiB `max_rpc_payload_bytes` and 40 MiB gRPC cap are proposed, nonbinding illustrations; both remain owner gate 5. Oversized worker-layer dispatches split deterministically into numbered chunks once approved limits are resolved.

### 7.2 `ExpertLayerRequest.v1`

One physical request is packed per `(target_worker_id, layer_id, chunk_index)` and contains explicit expert segments:

```text
protocol_version, schema_version
model_id, model_revision, weight_manifest_digest
run_id, cohort_id
worker_layer_dispatch_id, attempt_id
target_worker_id, expected_worker_incarnation
layer_id, phase=PREFILL|DECODE
dtype=BF16, hidden_dimension=2048
row_count, payload_bytes, deadline_budget_ns, traceparent
chunk_index, chunk_count
segments[]
activations: contiguous [row_count, 2048] BF16 bytes
payload_crc32c (optional, configured)
```

Each `ExpertSegment` is:

```text
logical_expert_batch_id
expert_id, replica_id, expected_expert_digest, residency_generation
activation_row_start, activation_row_count
rows[] = {
  route_id, request_id, absolute_token_position,
  flattened_token_row, topk_slot, padding_or_inactive
}
```

Segments are ordered by `(expert_id, replica_id, logical_expert_batch_id)`; rows within a segment follow the pinned reference token/top-k order. Duplicate activation bytes are allowed in v1 if one token selects multiple experts on one worker. Routing weights do **not** appear in requests and are never worker-owned.

### 7.3 `ExpertLayerResponse.v1`

```text
echoed protocol/schema/model/run/cohort/dispatch/attempt/layer/chunk fields
worker_id, worker_incarnation, residency_generation
segment_results[] = {logical_expert_batch_id, expert_id, replica_id,
                     expert_digest, route_ids[], status}
outputs: contiguous [row_count, 2048] BF16 unweighted bytes
worker_spans = {rpc_received, queue_enter, queue_start,
                host_to_device_start/end, compute_start/end,
                device_to_host_start/end, serialization_start/end,
                response_write}
terminal_status
```

Responses can complete in any order. Before acceptance, the coordinator validates every version, revision, digest, incarnation, generation, identity, cardinality, shape, dtype, and route ID. It reorders raw outputs into the pinned logical combination order, applies the original coordinator-retained routing weights, and accumulates once.

### 7.4 Proposed MVP inference API wire shape

The compatibility manifest versions the accepted public boundary independently of the internal RPC. The proposed minimum request fields are:

```text
ChatCompletionRequest.v1 = {
  model, messages[{role, content}], stream,
  max_completion_tokens, temperature, top_p, seed, stop,
  request_timeout_ms, client_request_id
}

CompletionRequest.v1 = {
  model, prompt, stream,
  max_tokens, temperature, top_p, seed, stop,
  request_timeout_ms, client_request_id
}
```

Unknown or unsupported fields return a versioned `unsupported_parameter` error; they are not ignored. Non-streaming responses expose stable `id`, `object`, `created`, `model`, ordered `choices`, terminal reason, and input/output token usage. Streaming uses ordered SSE `data:` chunks containing the stable response/request identity, choice index, incremental content, and optional terminal usage, followed by one terminal marker. Errors use `{error:{code,type,message,param,retryable}}`. Client disconnect or explicit cancellation maps to the request cancellation lifecycle in §6.3. The exact field names and parity obligations remain owner gate 4; this proposed shape cannot be advertised as full OpenAI compatibility until that gate is resolved and the compatibility suite passes.

## 8. End-to-end flows

### 8.1 Admission, prompt, and streaming

1. Validate the supported OpenAI-compatible subset and reject unsupported fields rather than ignoring them.
2. Resolve the pinned tokenizer/chat template and generation defaults; assign session/request/token identities.
3. Persist the pending run/factor identities, model/environment fingerprint, topology/placement/directory epochs, seed schedule, warm/cold cell, and deadline.
4. Enforce bounded request count, token budget, KV-memory budget, admission timeout, and maximum context.
5. Form a compatible fixed cohort by model, generation parameters, priority, and deadline class.
6. Stream SSE chunks with stable request IDs; cancellation propagates to future dispatches and attempts.

The MVP API supports `/v1/chat/completions`, `/v1/completions`, streaming, bounded concurrency, deadlines, and cancellation. Exact accepted request fields, SSE details, and concurrency limits are a project-owner gate; the service must publish a machine-readable compatibility document and cannot claim full OpenAI API parity.

### 8.2 Prefill

1. Left-pad a cohort to `[B,Smax]`, build attention/cache positions and a coordinate map, and mark padding/inactive rows.
2. Run one batched causal prefill with `use_cache=True`; all attention and KV work remains coordinator-local.
3. At every one of 48 layers, execute the per-layer flow below for all `[B,Smax]` rows; padding remains traceable and excluded from useful-token denominators.
4. Apply final normalization and LM head; select the first generated token from each request's final real prompt position.
5. Store KV state and stream the first token.

Correctness comparisons use the same batch shape, padding policy, and attention backend. Separate unbatched references are not substituted for a batched distributed cohort.

### 8.3 Per-layer expert dispatch

For each layer:

1. Receive post-attention/post-normalization hidden states and flatten `[B,S,2048]` to `[T,2048]`.
2. Compute full router logits `[T,128]`; record every score for every real token-layer.
3. Apply pinned FP32 softmax, exact top-8, selected-weight renormalization, and hidden-dtype cast.
4. Create `8*T_real` logical routes; retain routing weights and original row coordinates centrally.
5. Look up exact replicas from one immutable directory revision.
6. Seal compatible logical expert batches, invoke the scheduler, then pack selected batches by worker/layer/chunk.
7. Execute reused asynchronous unary RPC attempts under bounded queues and deadlines.
8. Workers may form compatible bounded kernel microbatches, execute exact resident MLPs, and return unweighted rows.
9. Validate and accept exactly one result per logical route; replan failed batches only to exact replicas.
10. Restore pinned logical row order, apply original weights, accumulate `[T,2048]`, and reshape.
11. Continue the unmodified decoder residual path.

The next layer cannot start for a cohort until every required current-layer output has succeeded or the request fails. That layer barrier is measured, not hidden.

### 8.4 Decode, completion, and persistence

For each generation step, pass the new cohort token column, masks, cache positions, and prior KV cache through the same outer loop used by the local reference. Finished rows remain inert until cohort completion in the Phase 1 proposal; cache compaction is gated. Greedy argmax is normative for correctness. Tokens are streamed per request until EOS, length limit, cancellation, deadline, or explicit failure.

After completion or failure, the run closes all logical terminal outcomes, writes raw events and artifacts, validates integrity, calculates content hashes, writes `status.json` and `manifest.json`, and atomically publishes `COMPLETE`. An incomplete attempt is retained but cannot be aggregated.

## 9. Placement, scheduling, and three batching stages

### 9.1 Offline placement study

Let exact expert `i=(l,e)` range over 48 layers and 128 experts. Before any policy executes, `constraint-set.v1` enumerates the immutable finite candidate universe `Q` as slots `q=(worker_id,tier,slot_ordinal)`, with unique canonical `candidate_slot_id = sha256(JCS(q))`. `slot_ordinal` is an unsigned integer scoped to one `(worker_id,tier)` and is never assigned or renumbered by a policy. `x[i,q] in {0,1}` means exact expert `i` occupies candidate slot `q`; `s_i` is its exact resident bytes, `C[w,t]` measured allocatable bytes for worker/tier before explicit reserve `R[w,t]`, `K[w,t]` the slot limit, `a[i,q]` the model/expert-digest/dtype/runtime/tier compatibility predicate, and `d(q)` the slot's declared failure domain. One slot holds at most one exact replica. A policy receives this universe read-only and may neither add, remove, relabel, reorder, nor alter a slot or constraint input.

Placement demand comes only from correctness-qualified `LIVE_E2E` routing traces partitioned before estimation. Near-duplicate source documents, conversations, and prompt templates form indivisible groups; a salted stable hash assigns each whole group, including all requests, repetitions, tokens, layers, and routes, to training, validation, or held-out test, with deterministic workload stratification that never breaks a group. Training estimates features; preregistered validation may tune hyperparameters, stopping, and comparator selection; test membership and payloads remain sealed until every candidate plan and comparator identity is hashed. Tokens from one request are correlated subsamples, not independent repetitions.

Statistics are records over the full Cartesian domains declared by the statistics spec, not sparse “observed-only” rows. Each record carries `statistic_kind`, `split`, `token_phase`, `workload_stratum`, canonical coordinates, `numerator`, `denominator`, `eligible_count`, `excluded_count`, `value_status=VALUE|NA`, and, for `NA`, a typed `na_reason`. For each split/phase/stratum, eligible observations are real non-padding tokens from correctness-qualified requests with a complete valid original-router record for the coordinate's required layer or layer pair; padding, invalid/incomplete routes, and tokens outside the stratum are counted as excluded under disjoint reason codes. Let `I[u,l,e]=1` when expert `e` is selected at layer `l`, and let `w[u,l,e]` be its normalized routing weight:

```text
f[l,e] = sum_u I[u,l,e] / N_l

m[l,e] = sum_u w[u,l,e] I[u,l,e]
         / sum_(u,e') w[u,l,e'] I[u,l,e']

c_same[l,e,e'] = sum_u I[u,l,e] I[u,l,e'] / N_l

lift_same[l,e,e'] =
  (joint_count[l,e,e'] * N_l) /
  (selection_count[l,e] * selection_count[l,e'])

c_cross[l,e,l+d,e'] = sum_u I[u,l,e] I[u,l+d,e'] / N_(l,l+d)
```

Here `N_l` is the eligible count for layer `l`; `N_(l,l+d)` counts only tokens with eligible observations at both layers. For `f`, the numerator is `selection_count=sum_u I` and denominator/eligible count are `N_l`. For `m`, the numerator is the selected normalized routing mass for `e`, the denominator is total selected normalized routing mass over all eligible `(u,e')`, and `eligible_count=N_l`. For `c_same`, emit only unordered expert pairs in the canonical domain `0 <= e < e' < 128`; never emit diagonal or symmetric duplicates. Its numerator is `joint_count` and its denominator/eligible count are `N_l`. The equivalent raw-count `lift_same` record publishes `numerator=joint_count*N_l` and `denominator=selection_count_e*selection_count_e'` over the same eligible population; it is `NA` with `ZERO_MARGINAL_FREQUENCY` if either marginal count is zero. For `c_cross`, configured offsets are a sorted unique set of positive `d`; emit ordered coordinates `(l,e,l+d,e')` only where `l+d<48`, with numerator the joint-selection count and denominator/eligible count `N_(l,l+d)`. The order of the two layer/expert coordinates is semantic and is not symmetrized.

An empty split/phase/stratum yields `NA(EMPTY_STRATUM)` for every statistic in its declared domain. Within a nonempty stratum, `N_l=0` yields `NA(NO_ELIGIBLE_LAYER_OBSERVATION)` for frequency, same-layer co-activation, and their lifts; `N_(l,l+d)=0` yields `NA(NO_ELIGIBLE_LAYER_PAIR)` for cross-layer co-activation; and a zero routing-mass denominator yields `NA(ZERO_ROUTING_MASS)`. Every `NA` retains its zero numerator/denominator and eligible/excluded counts; analyzers, policies, exports, and plots must never coerce `NA` to numeric zero. `activation-statistics.v1` publishes these fields, request-cluster bootstrap interval or its typed ineligibility, split provenance, workload stratum, and separate prefill/decode results. Selection frequency `f`, not routing-weight mass `m`, predicts activation rows and traffic capacity. Same-layer co-activation may reduce worker fan-out and packed RPC cost. Cross-layer co-activation is descriptive only and has zero direct distance weight because every layer returns through the coordinator; it cannot become a locality term without a later execution architecture that creates a causal worker-to-worker path.

For each permitted training or validation token-layer group `g`, the frozen surrogate scheduler maps selected experts to candidate resident slots using the same policy-independent canonical ordering as runtime `FixedPrimary`, the candidate assignment, frozen discovery/service snapshots, and no live state. Let `Q_g(x)` be the resulting required slot set, `B_request[g,q]` and `B_response[g,q]` its exact packed payload bytes/classes, and `R[g,q]` its rows. The normative transport term is one joint measured application-path round-trip surface `L_rpc[c,q](B_request,B_response)`, consumed exactly once for each `q in Q_g(x)`. Its exact lookup key is `(source_coordinator_id,candidate_slot_id,endpoint_id,transport,TLS_mode,request_payload_class,response_payload_class,request_bytes,response_bytes,concurrency_class,network_condition,latency_discovery_snapshot_sha256,uncertainty_coverage_cell_id)`; byte coordinates select an exact measured point or declared in-domain interpolation on the jointly sampled surface. It is derived directly from coordinator-monotonic round-trip observations of a no-expert-compute probe on the same data interface; it makes no one-way inference and is never split into outbound/inbound intercepts.

The worker term `S[q](rows,payload_class,concurrency,residency_state)` is selected from the content-addressed service-curve table in the frozen `calibration-set.v1`, keyed exactly by `(candidate_slot_id,worker_id,tier,hardware_revision,software_revision,model_revision,expert_class,dtype,rows,payload_class,concurrency_class,residency_state,service_calibration_snapshot_sha256)`. It includes the declared frozen queue/service estimator but no transport. For the MVP surrogate, fixed coordinator packing/staging cost is explicitly `P_coord=0 ns` in the hashed `objective-spec.v1`; observed packing/staging remains a separate runtime metric and is not silently allocated to either `L_rpc` or `S`. Thus the normative predicted critical remote cost is:

```text
C_hat_g(x) = max over q in Q_g(x) of
  [L_rpc[c,q](B_request[g,q], B_response[g,q])
   + S[q](R[g,q], payload_class[g,q], concurrency_class, residency_state)
   + P_coord]

minimize lexicographically:
  (Q_0.95(C_hat_g), mean(C_hat_g), mean(|workers(Q_g(x))|),
   replica_bytes(x), movement_bytes(x))
```

The first two components use only training estimators and preregistered validation estimators allowed by the policy specification—never sealed held-out test routes or outcomes. For immutable startup, `movement_bytes=0`; later transition studies may define it relative to a frozen prior assignment. `objective-spec.v1` content-addresses the formula, quantiles, integer units, payload classifier, `P_coord` rule, `latency-discovery-snapshot.v1` table/hash and exact key columns, `calibration-set.v1` service table/hash and exact key columns, interpolation domains, uncertainty/coverage gates, and missing-cell behavior. It also binds the permitted grouped-route artifact used to derive `Q_g/R/B`, the `constraint-set.v1` expert-size table used for `replica_bytes`, and either `movement_mode=IMMUTABLE_STARTUP_ZERO` or an explicit prior-plan hash used for `movement_bytes`; `mean(|workers(Q_g)|)` is recomputed from the same slot selections. Every lookup records its artifact hash, complete key, source row/cell ID, interpolation endpoints, and quantized returned value in the planning evidence. A required latency or service cell that is missing, stale, under-covered, outside the declared interpolation domain, or keyed to a different artifact returns structured `MISSING_OBJECTIVE_CELL`, `STALE_OBJECTIVE_CELL`, `UNDERCOVERED_OBJECTIVE_CELL`, or `OUT_OF_DOMAIN_OBJECTIVE_CELL` infeasibility. Extrapolation, configured-delay substitution, NA-to-zero coercion, and implementation-specific intercept allocation are forbidden. Consequently §23 can reproduce every objective tuple from sealed artifacts alone.

All objective inputs are quantized to declared integer units such as nanoseconds and bytes. The solver identity/version, seed, resource limits, candidate order, surrogate-scheduler identity, objective version, and quantization are frozen. Objective tuples compare lexicographically, then the canonical assignment vector over all `(i,q)` sorted by `(layer_id,expert_id,worker_id,tier_rank,slot_ordinal)` breaks any remaining tie. Identical inputs must emit byte-identical plan content. “Optimal” is allowed only with a verified certificate or valid bound; otherwise report the achieved tuple and deterministic stopping reason.

Every executable plan satisfies all hard constraints:

```text
r_min[i] <= sum_q x[i,q] <= r_max[i]
x[i,q] <= a[i,q]
sum_i x[i,q] <= 1
sum_(i,q:w(q)=w,t(q)=t) s_i*x[i,q] <= C[w,t] - R[w,t]
sum_(i,q:w(q)=w,t(q)=t) x[i,q] <= K[w,t]
sum_(q:d(q)=d) x[i,q] <= U[i,d]
sum_(q not in F) x[i,q] >= required_survivors[i,F] for every declared resilience failure set F
sum_(i,q) x[i,q] = exact_replica_count_budget
sum_(i,q) s_i*x[i,q] = exact_resident_byte_budget
```

`constraint-set.v1` is the sole common feasibility authority. It binds the complete candidate-slot inventory; `C`, `R`, and `K` for every worker/tier; exact expert sizes, dtype, tensor digests, runtime and slot compatibility matrix `a`; worker/tier/failure-domain mappings; `r_min/r_max`; `U`; resilience failure sets and required survivors; exact comparison replica-count and resident-byte budgets; and any transition-headroom rule. Each table is inline or has a content hash, and the root `constraint_set_sha256` is included in `placement-study-snapshot.v1`. It is required equal across every policy in a cell and is not nested in the policy-owned variation. Phase 2A binds the already selected one-copy budget; a Phase 2B constraint set remains unresolvable until the owner approves its exact common budgets. A policy attempt to provide a different constraint hash or mutated universe is `POLICY_MUTATED_FEASIBLE_UNIVERSE` and cannot produce a plan.

The feasibility proof also verifies unique deterministic `replica_id = sha256(JCS({model_revision,layer_id,expert_id,candidate_slot_id}))`, exact model/expert digest, runtime, dtype, tier, worker and bin-packing compatibility, per-expert min/max replication, failure-domain caps and declared resilience coverage, exact budgets, and transition headroom where applicable. Aggregate bytes are only a precheck. Missing required discovery/service cells, stale age, inadequate coverage/sample confidence/drift, insufficient capacity, or any constraint violation returns structured infeasibility with violated constraints and slacks; it never emits a partial executable plan or imputes configured delay.

Phase 1 requires only **manual explicit** placement, exercised across at least two distinct static configurations on identical hardware/topology/workload so that a measurable placement effect on latency and correctness is established before any optimization claim is attempted. Phase 2A fixes `r_min[i]=r_max[i]=1`, exactly 6,144 replicas, and compares these six versioned policies on one sealed snapshot to converge on an optimal placement strategy:

1. **Manual explicit:** operator-authored before test unsealing, with contamination disclosed.
2. **Seeded random feasible:** a versioned constructive feasible sampler over every preregistered seed, never a best-draw claim and not asserted uniform over feasible plans.
3. **Popularity-only:** training selection frequency plus hard capacity, without topology costs, service curves, co-activation, validation/test routes, or live state.
4. **Topology-only/uniform-demand:** frozen directional topology/discovery and capacity with a uniform expert-demand prior, without popularity or co-activation.
5. **Co-activation-aware:** permitted training popularity and same-layer co-activation plus frozen topology to reduce fan-out; cross-layer statistics remain descriptive.
6. **Optimized candidate:** the complete permitted training feature set, with hyperparameters/comparator selected only on preregistered validation.

All six are required before claiming policy superiority. `placement-plan.v2` contains only desired assignments/tiers/digests/replica IDs, input hashes, solver identity, objective components, feasibility-proof hash, and the canonical tie rule. It must reject fields encoding primary or fallback order, queue/health scores, per-request choices, attempts, dispatch decisions, or scheduling. Phase 2B may study bounded static replication only after the project approves an exact common replica-count and resident-byte budget variable for the cell; this document selects no concrete Phase 2B budget.

### 9.2 Runtime replica selection

`FixedPrimary` is the Phase 1 and placement-study scheduler: among eligible replicas for an exact expert, it selects the lexicographically least canonical key `(worker_id UTF-8 bytes,tier_rank from constraint-set.v1,slot_ordinal unsigned integer,replica_id bytes)`. `scheduler-spec.v1` hashes `candidate_order_version`, that complete key, byte/case/integer encoding rules, eligibility predicate, and `tie_rule=LEXICOGRAPHIC_MIN`; neither plan-row order, serialization order, directory insertion order, nor policy output order participates. The offline surrogate scheduler references the same `scheduler_spec_sha256` and applies the identical ordering. In the one-copy Phase 2A cells its choice is trivial. `StableHash` and `LeastPredictedFinish` are separate scheduler-study policies. The latter may score the joint frozen RPC surface, runtime-observed queue work, calibrated service time, cold-load estimate, and reliability penalty, recording every component and tie-breaking by the same canonical key.

The runtime scheduler may consume health, queues, attempts, and fallback rules, but it cannot modify selected expert IDs or desired placement. Missing replicas yield `NO_REPLICA`. The offline placement policy cannot inspect these runtime inputs or select a dispatch replica. Replication and alternate-scheduler experiments remain separate factors.

### 9.3 Batching stages

The three stages are separate, versioned, and measured:

1. **Cohort batching:** API requests with compatible generation parameters; one padded prefill and synchronous decode membership.
2. **Dispatch packing:** logical route rows become expert segments after replica choice, then segments become `(worker,layer,chunk)` RPCs.
3. **Worker microbatching:** admitted compatible rows may coalesce into GPU launches by `(model_revision,layer_id,expert_id,replica_id,dtype,hidden_dimension,token_phase)`.

A dispatch batch seals on the first of item, byte, wait, earliest-deadline guard, or end-of-layer limits. Worker admission is atomic and bounded by ingress bytes, queued batches/rows, pinned buffers, and GPU slots. Batching never changes routing, placement, or replica selection. Placement/scheduler/topology studies freeze `batching_id`; batching experiments declare `vary_only=batching`.

Metrics distinguish logical expert accesses, logical expert batches, packed RPCs, RPC attempts including retries, and worker kernel microbatches.

## 10. Topology, physical path, heterogeneity, and faults

Every ordered data-plane node pair has its own link record. `A->B` and `B->A` do not inherit from each other. Canonical delay is one-way; an accepted symmetric RTT shorthand must be expanded before hashing/execution. Missing links fail validation. Physical site and logical region are independent fields. Declared/vendor hardware capabilities and measured expert profiles are separate namespaces.

Placement uses a first-class `latency-discovery-snapshot.v1`, never configured link values. Each application-path gRPC round-trip cell is keyed by `(source_coordinator_id,candidate_slot_id,endpoint_id,transport,TLS_mode,request_payload_class,response_payload_class,request_bytes,response_bytes,concurrency_class,network_condition,uncertainty_coverage_cell_id)` and uses the same TCP/gRPC data interface as expert traffic with a no-expert-compute response handler. Payload classes include control probes and representative packed activation/response size pairs; the declared interpolation domain is joint in request and response bytes. The content-addressed snapshot binds its discovery spec, measurement window, environment/software fingerprints, candidate-slot inventory hash, topology hash, optional topology-proof hash, required-cell set, and for every cell: successful/failed sample counts, timeout/loss, p50/p95/p99 coordinator-monotonic round-trip observations, request-cluster bootstrap bounds, relative interval width, early-versus-late drift, coverage, interpolation bounds, and age at plan/evaluation admission.

```text
coverage = count(valid required cells) / count(required cells)
age_at_admission = admission_time - measurement_window_end
source_kind = ACTIVE_MEASURED
network_condition = PHYSICAL_UNDERLAY | EMULATED_PROFILE
```

Required coverage, minimum sample count, maximum interval width, drift, and age are preregistered owner-gated thresholds. Any failure makes the study snapshot infeasible; no missing measurement is filled from `topology-spec.v1`. An active measurement under emulation is labeled “measured under the named applied emulation profile,” not physical-underlay or WAN evidence. Probes run in a declared quiescent pre/post window unless concurrent probing is the named treatment, so discovery does not become an unrecorded workload. Source/coordinator-monotonic joint round trips are authoritative and are used once as `L_rpc`; request and response payload classes remain separate key dimensions but are not converted into additive one-way costs. One-way latency is never inferred without the clock gate in §12.2.

For `A->B`, the proposed packet path is:

```text
A grpc process -> A data-interface egress HTB/source-uplink class
 -> physical/overlay network
 -> B ingress clsact/mirred -> B IFB HTB/downlink+directional class
 -> netem delay/jitter/loss -> B TCP/grpc stack
```

Linux shaping is egress-oriented; IFB plus `mirred` makes ingress shapeable ([tc-mirred](https://man7.org/linux/man-pages/man8/tc-mirred.8.html)). HTB supplies hierarchical rates ([tc-htb](https://man7.org/linux/man-pages/man8/tc-htb.8.html)); netem supplies delay, jitter, loss models, rates, and seeds while documenting timer/TCP limitations ([tc-netem](https://man7.org/linux/man-pages/man8/tc-netem.8.html)). Delay/loss is placed at receiver ingress to reduce sender-egress TCP artifacts. The controller applies rules before admission, reads back `tc -j -s`, hashes applied state, runs application-level directional probes, and opens the benchmark gate only after validation.

Where a leg's actual RPC path runs on a macOS host rather than Linux, an equivalent directional shaping mechanism must supply the same before/after read-back and probe-validated guarantee — candidates are Apple's Network Link Conditioner profiles, or `pf` combined with `dnctl`/`ipfw` dummynet. The exact macOS mechanism, its rule/counter read-back format, and its topology-proof evidence mapping remain an explicit Phase 1 gate, not a default; `topology-proof.v1`'s `tc_htb_ifb_netem_readback` field is Linux-specific and needs a parallel macOS evidence field before a mixed Mac/Linux topology can pass topology proof.

Source uplink, directional link, and destination downlink are distinct capacities. Management, health, and telemetry traffic use an unshaped management address; only the data RPC binds the shaped address. Application sleeps do not emulate the network.

Faults include link blackhole/reset/slowdown/loss, worker terminate/pause/restart, expert unload/failure, cold loading, queue saturation, and explicit modeled compute delay. Each emits scheduled time, apply start/end, observed time, mechanism, and outcome. A restarted worker gets a new incarnation. A delay failpoint is labeled modeled slowdown, never measured GPU performance.

Network emulation is evidence about the applied profile and tested stack, not a faithful named WAN. A real multi-region claim requires a separate physical deployment and validation gate.

Five authorities remain distinct: `topology-spec.v1` configures intended impairment and logical regions; `latency-discovery-snapshot.v1` measures the named application path during a bounded window; logical regions classify experiments; validated physical-site/link evidence classifies physical WAN/non-WAN; and only a paired delay-only control with all other factors frozen estimates RTT contribution. None substitutes for another.

## 11. Complete versioned configuration contracts

All schemas reject unknown required-semantic fields, carry `schema_version`, are resolved to canonical JSON for hashing, and are stored verbatim with the run.

| Object/schema | Required fields |
|---|---|
| `model-spec.v1` | model/revision/index/tokenizer/chat/generation/runtime/precision/quantization/attention/shared strategy/router semantics/context |
| `workload-spec.v1` | prompt manifest hash/source/license class, arrival model, concurrency, lengths, generation, request count, warm/cold |
| `hardware-inventory.v1` | profiles, node bindings, CPU/accelerator (CUDA GPU or Apple MPS)/RAM/VRAM or unified memory/NIC, usable capacities, declared performance, measured profile references |
| `topology-spec.v1` | physical sites, logical regions, LANs, nodes/addresses, source/downlink caps, every ordered link and impairment |
| `latency-discovery-snapshot.v1` | discovery spec/hash, joint application-path RPC keys/surfaces/interpolation domain, candidate-slot inventory hash, window, counts/distributions/bounds/drift/loss, coverage/age, network-condition authority, topology/proof hashes |
| `routing-partition-manifest.v1` | grouped split algorithm/salt commitment, train/validation/test IDs and trace hashes, strata counts, duplicate/leakage audit, test seal/unseal events |
| `activation-statistics.v1` | split/phase/stratum provenance, full statistic domains, numerator/denominator/eligible/excluded/value-or-typed-NA fields, pair/offset rules, request-cluster intervals, analyzer identity |
| `constraint-set.v1` | immutable candidate slots; worker/tier capacity/reserve/slot inputs; expert sizes/digests/dtype/runtime compatibility; failure domains; replica bounds/caps; resilience sets; exact budgets; transition headroom; hashes for every table |
| `objective-spec.v1` | joint `L_rpc` and `S` artifact hashes/exact keys/domains, payload classifier, explicit coordinator-cost rule, quantization, formula/quantiles, fail-closed errors |
| `placement-study-snapshot.v1` | sealed model/hardware/topology/discovery/calibration/constraint/objective/scheduler/partition/statistics hashes, feature-access policy, freshness policy, test-seal commitment |
| `placement-policy-spec.v1` | policy source/executable hash, feature allowlist/mask, common constraint/objective/surrogate-scheduler references, solver/version/seed/limits, quantization/ties, `online_updates=FORBIDDEN` |
| `placement-plan.v2` | snapshot/policy/common constraint/objective/scheduler hashes, placement epoch, complete assignment/replica/digest/candidate-slot rows, exact budgets, objective components/lookups, feasibility-proof hash, tie rule; no runtime selection fields |
| `placement-feasibility-proof.v1` | common constraint hash, per-expert/per-slot/per-worker-tier/budget/failure-domain checks and slacks, universe-mutation audit, certificate/bound if valid, structured infeasibility |
| `placement-comparison-contract.v1` | cell/roster/contrasts, required-equal common constraint/objective/scheduler and other fields, permitted derived differences, exact budget, paired order, reset/readiness/repetitions/statistics, invalidation and claims |
| `start-state-record.v1` | reset procedure, cache/channel/residency/readiness state, warmup/stability proof, drift checks, semantic equivalence |
| `placement-application-record.v1` | plan/directory hashes, startup state transitions, load/digest/capacity/residency/incarnation checks, atomic commit or pre-commit abort, readiness |
| `directory-snapshot.v1` | committed revision, placement epoch, worker incarnations, replica residency/health/generation, snapshot time |
| `scheduler-spec.v1` | policy/version/config, eligibility, snapshot age, retries, deadlines, seed domain, canonical candidate-order version/key/encodings/tie rule |
| `batching-spec.v1` | separate cohort, dispatch packing, and worker microbatch limits/compatibility |
| `fault-plan.v1` | scheduled target/action/mechanism/duration/seed |
| `execution-mode-spec.v1` | primary mode, fidelity vector, claim allowlist version |
| `calibration-set.v1` | immutable service-calibration snapshot and `S` curves with exact keys/domains, bound discovery snapshot, explicit measured/configured authority, environment/hardware keys |
| `replay-dependency-edge.v1` | source/destination logical event IDs, dependency kind, release rule, source trace/run hashes |
| `baseline-comparison.v1` | reference/candidate manifest hashes, preregistered tolerance identity/time, equal/permitted-different fields, observations |
| `topology-proof.v1` | run/topology hashes, applied rule bindings, counters, directional probes and outcomes |
| `validation-record.v1` | analyzer identity, immutable evidence hashes, completeness/topology/baseline predicates, claim matrix, validity |
| `run-manifest.v1` | schema-bundle/identity-registry hash and global rule; every recomputed factor hash; study/partition/statistics/policy/plan/feasibility/comparison/start-state/application hashes; initial directory; cell/pair/order; `vary_only`; test-unseal event; seeds/warm state/software/output/security |

The schema bundle contains this closed immediate artifact-identity registry. `none` means the artifact ID is out of band in the immutable artifact index/parent reference and the complete object is hashed; it does not mean “unhashed.” Exactly 57 immediate types appear once below: the 31 revision-6 types plus 26 audited child/evidence types. Every row has exactly one authoritative schema source owned by that same `schema-bundle.v1`; the core-root table above is a summary, not a substitute for the audited child schemas. The four non-`none` entries preserve the intentional embedded identifiers already used by analyzer/replay records. A new artifact type cannot be emitted until exactly one registry row is added, and registry construction rejects missing types, duplicate type rows, aliases, and more than one candidate own field.

| Artifact type | `own_identity_field` |
|---|---|
| `artifact-identity-registry` | `none` |
| `experiment-bundle.v1` | `none` |
| `model-spec.v1` | `none` |
| `workload-spec.v1` | `none` |
| `hardware-inventory.v1` | `none` |
| `topology-spec.v1` | `none` |
| `latency-discovery-snapshot.v1` | `none` |
| `routing-partition-manifest.v1` | `none` |
| `activation-statistics.v1` | `none` |
| `constraint-set.v1` | `none` |
| `objective-spec.v1` | `none` |
| `placement-study-snapshot.v1` | `none` |
| `placement-policy-spec.v1` | `none` |
| `placement-plan.v2` | `none` |
| `placement-feasibility-proof.v1` | `none` |
| `placement-comparison-contract.v1` | `none` |
| `start-state-record.v1` | `none` |
| `placement-application-record.v1` | `none` |
| `directory-snapshot.v1` | `none` |
| `scheduler-spec.v1` | `none` |
| `batching-spec.v1` | `none` |
| `fault-plan.v1` | `none` |
| `execution-mode-spec.v1` | `none` |
| `calibration-set.v1` | `none` |
| `event-priority-table.v1` | `none` |
| `replay-completeness-profile.v1` | `none` |
| `replay-dependency-edge.v1` | `edge_id` |
| `baseline-comparison.v1` | `comparison_id` |
| `topology-proof.v1` | `topology_proof_id` |
| `validation-record.v1` | `validation_record_id` |
| `run-manifest.v1` | `none` |
| `schema-bundle.v1` | `none` |
| `api-compatibility-manifest.v1` | `none` |
| `claim-rules.v1` | `none` |
| `validation-completeness-rules.v1` | `none` |
| `raw-event-manifest.v1` | `none` |
| `physical-site-link-inventory.v1` | `none` |
| `tolerance-spec.v1` | `none` |
| `latency-discovery-spec.v1` | `none` |
| `interpolation-domain.v1` | `none` |
| `measurement-admission-policy.v1` | `none` |
| `candidate-slot-inventory.v1` | `none` |
| `worker-tier-limits.v1` | `none` |
| `expert-contracts.v1` | `none` |
| `expert-slot-compatibility-matrix.v1` | `none` |
| `failure-domain-inventory.v1` | `none` |
| `replica-bounds-and-caps.v1` | `none` |
| `resilience-failure-sets.v1` | `none` |
| `solver-limits.v1` | `none` |
| `service-calibration-snapshot.v1` | `none` |
| `directory-replica-records.v1` | `none` |
| `trace-manifest.v1` | `none` |
| `attempt-status.v1` | `none` |
| `metric-summary.v1` | `none` |
| `environment-fingerprint.v1` | `none` |
| `security-policy.v1` | `none` |
| `retention-policy.v1` | `none` |

Accordingly, generic factors such as latency discovery, constraint set, objective spec, placement-study snapshot, scheduler spec, calibration set, plan, proof, directory, start/application records, and run manifest do not embed their own digest. Fields such as `constraint_set_sha256` inside a placement snapshot or plan and `latency_discovery_snapshot_sha256` inside calibration/objective objects are referenced child identities and remain in the parent's hash projection. The four registered analyzer/replay records embed exactly their listed field and no alias.

Seven semantically parsed constraint leaves—`candidate-slot-inventory.v1`, `worker-tier-limits.v1`, `expert-contracts.v1`, `expert-slot-compatibility-matrix.v1`, `failure-domain-inventory.v1`, `replica-bounds-and-caps.v1`, and `resilience-failure-sets.v1`—remain registered while stored at separate URIs. A future revision may remove one only by fully inlining its complete object into `constraint-set.v1` and removing its separate URI and hash in the same resolved schema; retaining either makes the registry row mandatory.

Two conditional contracts are declared but are not part of the immediate count and are not activated by this design:

| Conditional type | `own_identity_field` | Primary parent when activated | Required before | Current state |
|---|---|---|---|---|
| `quality-evaluation-record.v1` | `none` | `validation-record.v1` | any modified/forced-routing quality evidence is emitted | `GATED_NOT_EMITTED` |
| `deletion-audit-record.v1` | `none` | `validation-record.v1` | any retention/deletion audit evidence is emitted | `GATED_NOT_EMITTED` |

Before either feature emits a file, its conditional row becomes an immediate row in a new schema-bundle/registry version and the complete closure validation reruns. Registering the identity shape selects no quality protocol, security policy, retention duration, or deletion action.

The immediate structured graph has the following unique primary-parent ledger. “Primary parent” is the single `parent_uri` owner used by `ARTIFACTS.sha256`; additional typed references are allowed but do not create duplicate inventory rows. The counts sum to 57 and every immediate type appears exactly once.

| Primary parent | Count | Immediate structured types owned exactly once |
|---|---:|---|
| detached structured roots | 2 | `run-manifest.v1`, `validation-record.v1` |
| `validation-record.v1` | 7 | `schema-bundle.v1`, `claim-rules.v1`, `validation-completeness-rules.v1`, `raw-event-manifest.v1`, `topology-proof.v1`, `baseline-comparison.v1`, `metric-summary.v1` |
| `run-manifest.v1` | 23 | `experiment-bundle.v1`, `api-compatibility-manifest.v1`, `model-spec.v1`, `workload-spec.v1`, `hardware-inventory.v1`, `topology-spec.v1`, `placement-study-snapshot.v1`, `placement-policy-spec.v1`, `placement-plan.v2`, `placement-feasibility-proof.v1`, `placement-comparison-contract.v1`, `start-state-record.v1`, `placement-application-record.v1`, `directory-snapshot.v1`, `scheduler-spec.v1`, `batching-spec.v1`, `fault-plan.v1`, `execution-mode-spec.v1`, `calibration-set.v1`, `attempt-status.v1`, `trace-manifest.v1`, `security-policy.v1`, `retention-policy.v1` |
| `schema-bundle.v1` | 1 | `artifact-identity-registry` |
| `placement-study-snapshot.v1` | 5 | `latency-discovery-snapshot.v1`, `routing-partition-manifest.v1`, `activation-statistics.v1`, `constraint-set.v1`, `objective-spec.v1` |
| `latency-discovery-snapshot.v1` | 3 | `latency-discovery-spec.v1`, `interpolation-domain.v1`, `measurement-admission-policy.v1` |
| `constraint-set.v1` | 7 | `candidate-slot-inventory.v1`, `worker-tier-limits.v1`, `expert-contracts.v1`, `expert-slot-compatibility-matrix.v1`, `failure-domain-inventory.v1`, `replica-bounds-and-caps.v1`, `resilience-failure-sets.v1` |
| `placement-policy-spec.v1` | 1 | `solver-limits.v1` |
| `calibration-set.v1` | 2 | `service-calibration-snapshot.v1`, `environment-fingerprint.v1` |
| `directory-snapshot.v1` | 1 | `directory-replica-records.v1` |
| `execution-mode-spec.v1` | 2 | `event-priority-table.v1`, `replay-completeness-profile.v1` |
| `trace-manifest.v1` | 1 | `replay-dependency-edge.v1` rows |
| `topology-proof.v1` | 1 | `physical-site-link-inventory.v1` |
| `baseline-comparison.v1` | 1 | `tolerance-spec.v1` |
| **Total** | **57** | complete immediate structured graph |

### 11.1 Raw/native identity declarations and exclusions

The following 20 declarations are byte/native identities, not JCS registry rows. Each indexed identity records its declaration ID in `ARTIFACTS.sha256`. Where a declaration lists multiple eligible parent types, each concrete instance selects exactly one as its primary `parent_uri`; every other relationship is a typed reference and never a second index row. `RAW_SHA256` means SHA-256 of exact stored bytes with no newline, Unicode, JSON, compression, Parquet, or protobuf normalization, plus their exact unsigned file length. `NATIVE_DIGEST` is one native identity with no byte-count assertion. If claim-relevant bytes are retained, the native identity and retained file are two explicitly referenced, distinct-URI index rows under the same owning structured parent; the retained row is `RAW_SHA256`, never a suffix or second value in the native row.

| Declaration | Exact representation | Media / encoding | Parser or schema authority | Parent binding | Digest and byte count | Required mutation fixture |
|---|---|---|---|---|---|---|
| `RAW-01_SCHEMA_SOURCES` | each schema source file at its normalized relative POSIX path; no concatenation | declared JSON Schema/protobuf/text media type; exact bytes | pinned JSON-Schema metaschema or compiler/tool version | `schema-bundle.v1` | `RAW_SHA256` per file + `byte_count`; bundle binds ordered path/digest/count rows | flip one byte in each media class → file and bundle IDs change |
| `RAW-02_ARTIFACT_INDEX` | canonical TSV, UTF-8, LF, no BOM, rows bytewise sorted by normalized URI | `text/tab-separated-values; charset=utf-8` | §17 fixed six-column grammar | detached publication-root tuple, the sole non-file trust anchor | `RAW_SHA256` + `byte_count`; never self-listed or self-hashed | flip byte/reorder row/change LF → root verification fails |
| `RAW-03_COMPLETE_MARKER` | exactly ASCII `COMPLETE\n` (9 bytes) | `text/plain; charset=us-ascii` | literal-byte comparator | `attempt-status.v1`; one `ARTIFACTS.sha256` row | fixed `RAW_SHA256` + `byte_count=9` | any one-byte edit/addition/removal fails publication |
| `RAW-04_EXTERNAL_MODEL_METADATA` | exact retrieved config/index/tokenizer/generation metadata bytes plus source URI/revision | source-declared media; no normalization | pinned upstream parser/revision in `model-spec.v1` | `model-spec.v1` | native provider revision row and, when retained, a distinct `RAW_SHA256` byte row | mutate either native revision or retained byte → its row and model ID change |
| `RAW-05_CHECKPOINT_SHARDS` | exact safetensors shard bytes in index order | `application/octet-stream` / safetensors | pinned safetensors format/parser and index | `model-spec.v1` and `expert-contracts.v1` | `RAW_SHA256` per shard + `byte_count` | flip one shard byte → tensor inventory/digest validation fails |
| `RAW-06_SOURCE_EXECUTABLES` | exact Git object/archive, source, solver, analyzer, or executable bytes | declared source/binary media | Git object rules and recorded build/toolchain | policy specs, feasibility proof, validation record | one Git/native identity row and a distinct retained `RAW_SHA256` row when bytes are kept | mutate native reference or retained source/binary byte → its row and parent ID change |
| `RAW-07_OCI_IMAGES` | OCI manifest/config/layer identities under the OCI Image Specification | OCI media types | pinned OCI distribution/image-spec version | environment/model/calibration/validation objects | one OCI `NATIVE_DIGEST` row and a distinct retained-manifest `RAW_SHA256` row when kept | mutate native manifest identity or retained manifest/config/layer byte → verification fails |
| `RAW-08_PROMPT_JSONL` | exact prompt JSONL bytes; one UTF-8 JSON value per LF-terminated line | `application/x-ndjson; charset=utf-8` | workload-declared JSONL record schema | `workload-spec.v1` | `RAW_SHA256` + `byte_count`; semantic rows are not opaque JSON objects | mutate whitespace/content/newline → workload ID changes |
| `RAW-09_COMMITMENTS_SALTS` | exact commitment preimage/salt bytes when retained; otherwise exact commitment identity | `application/octet-stream` | commitment algorithm/version in partition/spec parent | routing partition or placement-study snapshot | exact bytes use one `RAW_SHA256` row; a native commitment identity and retained bytes use two distinct rows | mutate identity or flip one retained byte → commitment verification fails |
| `RAW-10_TC_TOOL_EVIDENCE` | exact `tc -j -s`, command/tool/readback, and probe-support files | declared JSON or `text/plain`; exact bytes | recorded Linux `tc`/tool version; semantic JSON parsed by `topology-proof.v1` | `topology-proof.v1` | `RAW_SHA256` + `byte_count` | mutate counter/rule byte → topology proof fails |
| `RAW-11_DISCOVERY_PARQUET` | exact discovery-cells Parquet file bytes | `application/vnd.apache.parquet` | Arrow schema fingerprint bound by `latency-discovery-snapshot.v1` | `latency-discovery-snapshot.v1` | `RAW_SHA256` + `byte_count` | one-byte mutation or schema/count drift fails snapshot |
| `RAW-12_ACTIVATION_STATS_PARQUET` | exact activation-statistics Parquet bytes | `application/vnd.apache.parquet` | Arrow schema/domain fingerprint in `activation-statistics.v1` | `activation-statistics.v1` | `RAW_SHA256` + `byte_count` | one-byte mutation fails statistics recomputation |
| `RAW-13_SERVICE_CURVES_PARQUET` | exact service-curve Parquet bytes | `application/vnd.apache.parquet` | Arrow schema/key/domain fingerprint in `service-calibration-snapshot.v1` | `service-calibration-snapshot.v1` | `RAW_SHA256` + `byte_count` | one-byte mutation fails objective lookup verification |
| `RAW-14_TOPOLOGY_PROBES_PARQUET` | exact topology-probe Parquet bytes | `application/vnd.apache.parquet` | Arrow schema fingerprint in calibration/topology proof | `calibration-set.v1` and `topology-proof.v1` | `RAW_SHA256` + `byte_count` | one-byte mutation fails proof/calibration parents |
| `RAW-15_EVENT_PARQUET` | exact partitioned raw-event Parquet file bytes | `application/vnd.apache.parquet` | event schema bundle + partition-key contract | `raw-event-manifest.v1` | `RAW_SHA256` per partition + `byte_count` | one-byte mutation, missing/duplicate partition, or row-count drift fails manifest |
| `RAW-16_TRACE_PARQUET` | exact routing/logical-dispatch trace Parquet bytes | `application/vnd.apache.parquet` | trace schema fingerprint in `trace-manifest.v1` | `trace-manifest.v1` | `RAW_SHA256` + `byte_count` | one-byte mutation or coordinate/count drift fails trace |
| `RAW-17_REPLAY_EDGE_PARQUET` | exact Parquet container holding replay-edge rows | `application/vnd.apache.parquet` | container Arrow schema plus registered `replay-dependency-edge.v1` row projection | `trace-manifest.v1` / replay source manifest | container `RAW_SHA256` + `byte_count`; row `edge_id`s remain distinct | mutate container byte or row/order/schema → container or row verification fails |
| `RAW-18_TENSOR_PROTOBUF_BLOBS` | exact tensor capture and protobuf payload bytes with framing metadata | `application/octet-stream` or `application/x-protobuf` | pinned tensor/protobuf schema, dtype, shape, endian, framing | trace/evidence manifest | `RAW_SHA256` + `byte_count` | one-byte payload/framing mutation fails decode/digest checks |
| `RAW-19_METRIC_VISUAL_OUTPUTS` | exact Parquet/CSV/PNG/SVG output bytes; excludes semantic `metric-summary.v1` JSON | declared output media; exact bytes | metric schema or renderer/version bound by `metric-summary.v1` | `metric-summary.v1` | `RAW_SHA256` + `byte_count` | one-byte/table/render mutation invalidates derived output only, never raw evidence |
| `RAW-20_LOGS_DIAGNOSTICS` | exact log/diagnostic text or binary files | declared text/binary media; exact bytes | producer/version; no claim authority unless cited by structured evidence | `attempt-status.v1` or explicit evidence record | `RAW_SHA256` + `byte_count` | one-byte mutation invalidates cited parent; uncited logs remain non-authoritative |

The schema bundle's v1 native-algorithm enum is finite and inline, not a new artifact or service. An artifact uses one compatible token below or is indexed only as exact `raw_sha256` bytes; adding an algorithm requires a new reviewed schema-bundle version.

| Registered algorithm token | Compatible declarations | Canonical lowercase native-digest grammar after the first colon |
|---|---|---|
| `git-sha1` | `RAW-04_EXTERNAL_MODEL_METADATA`, `RAW-06_SOURCE_EXECUTABLES` | exactly 40 lowercase hexadecimal characters |
| `git-sha256` | `RAW-06_SOURCE_EXECUTABLES` | exactly 64 lowercase hexadecimal characters |
| `oci` | `RAW-07_OCI_IMAGES` | literal `sha256:` followed by exactly 64 lowercase hexadecimal characters |
| `commitment-sha256` | `RAW-09_COMMITMENTS_SALTS` | exactly 64 lowercase hexadecimal characters |
| `tensor-sha256` | `RAW-18_TENSOR_PROTOBUF_BLOBS` | exactly 64 lowercase hexadecimal characters |

Inline objects inherit the identity and schema of their registered structured parent and receive no second registry or raw declaration. A semantic JSON/YAML child at a separate URI must be registered; it may not be labeled opaque or `RAW_SHA256` to avoid JCS. Authoring YAML is never emitted as resolved evidence. Raw schema files are children of exactly one `schema-bundle.v1`. Registered `replay-dependency-edge.v1` row IDs and the `RAW-17` Parquet container identity are intentionally different levels and both are verified. Telemetry rows inherit `RAW-15`; they are not separately registered. The detached publication-root tuple `(attempt URI, index URI, digest, byte count)` is supplied to verification out of band because a raw index cannot hash a line containing its own digest; it accounts for the index exactly once without a self-reference and is not a new stored artifact or service.

These five runtime-only families are excluded unless a later schema persists one as standalone evidence, at which point closure must classify it before emission:

| Runtime-only family | Identity class | Why excluded now |
|---|---|---|
| public API request/response/SSE messages | `W0` | wire messages are represented by registered events or captures, not standalone files |
| expert RPC request/response messages | `W0` | live protobuf/tensor messages are runtime traffic; persisted captures use `RAW-18` |
| telemetry event rows inside persisted partitions | `W0` within `RAW-15` | row semantics belong to the event schema and raw-event manifest/container |
| domain semantic IDs (slot, replica, run, event, request, span, route) | `W0` semantic key | they remain hashed fields in parents but are not artifact identities |
| ephemeral queues, health, caches, channels, and in-memory directory views | `W0` | only typed persisted observations/snapshots enter the artifact graph |

The closure scanner freezes the following 77 current hash/digest/artifact-key names. A name may occur many times, but it has exactly one resolution class below; each concrete reference still resolves to one indexed child. Any new hash field, digest/key name, evidence kind, path class, or schema fails `IDENTITY_LEDGER_UNCLASSIFIED` until a reviewed registry/declaration update classifies it.

| Resolution class | Count | Exhaustive names | Resolution |
|---|---:|---|---|
| structured JCS references | 40 | `activation_statistics_sha256`, `api_compatibility_manifest_sha256`, `artifact_identity_registry_sha256`, `calibration_set_sha256`, `candidate_slot_inventory_sha256`, `comparison_contract_sha256`, `compatibility_matrix_sha256`, `configured_emulation_sha256`, `constraint_set_sha256`, `directory_snapshot_sha256`, `environment_fingerprint_sha256`, `failure_domains_sha256`, `feasibility_proof_sha256`, `freshness_and_coverage_policy_sha256`, `hardware_inventory_sha256`, `interpolation_domain_sha256`, `latency_discovery_snapshot_sha256`, `limits_sha256`, `objective_spec_sha256`, `placement_application_record_sha256`, `placement_plan_sha256`, `placement_policy_spec_sha256`, `placement_study_snapshot_sha256`, `prior_plan_sha256`, `replica_bounds_and_caps_sha256`, `replica_records_sha256`, `resilience_failure_sets_sha256`, `routing_partition_manifest_sha256`, `run_manifest_sha256`, `scheduler_spec_sha256`, `schema_bundle_sha256`, `service_calibration_snapshot_sha256`, `source_partition_sha256`, `source_run_manifest_sha256`, `source_trace_manifest_sha256`, `start_state_record_sha256`, `surrogate_scheduler_sha256`, `tolerance_artifact_sha256`, `topology_proof_sha256`, `topology_spec_sha256` | exactly one immediate/activated-conditional registry type and J0/J1 projection |
| raw SHA-256 references | 13 | `binary_sha256`, `cells_sha256`, `curves_sha256`, `data_sha256`, `grouped_route_source_sha256`, `prompt_manifest_sha256`, `readback_sha256`, `safetensors_index_sha256`, `salt_commitment_sha256`, `test_seal_commitment_sha256`, `test_trace_commitment_sha256`, `train_trace_sha256`, `validation_trace_sha256` | one `RAW-01`–`RAW-20` declaration and exact-byte index entry |
| polymorphic indexed references | 6 | `artifact_sha256`, `evidence_sha256`, `manifest_sha256`, `sha256`, `snapshot_sha256`, `source_sha256` | enclosing schema/declaration plus normalized URI names the expected index kind and type/declaration; never guessed from the field name or accepted bare |
| native/semantic digests | 6 | `container_digest`, `expected_expert_digest`, `expert_digest`, `image_digest`, `tensor_digest`, `weight_manifest_digest` | parent schema resolves `RAW-04`–`RAW-07` or `RAW-18`; the four container fields are typed `RAW-07` native locators and any retained bytes use a distinct raw locator |
| inline value hashes | 2 | `candidate_value_hash`, `reference_value_hash` | canonical scalar/value projection inside `baseline-comparison.v1`; not a standalone artifact or second registration |
| artifact identity/evidence keys | 8 | `artifact_id`, `comparison_id`, `edge_id`, `physical_link_evidence_id`, `retention_policy_id`, `security_policy_id`, `topology_proof_id`, `validation_record_id` | global J0/J1 identity, registered policy/evidence reference, or out-of-band alias of its registered structured type |
| aggregate indexed-reference collections | 2 | `factor_hashes`, `input_hashes` | every member is a typed normalized-URI locator for exactly one index row; a digest-only member is invalid |
| **Total** | **77** | no unclassified name | closure invariant |

The three polymorphic/aggregate forms are closed rather than weakly typed. A literal `sha256` occurs only inside a registered parent or declared evidence record that also supplies a normalized artifact `uri` and an unambiguous expected `identity_kind` plus registered type or raw/native declaration; its value must equal the SHA-256 component of that exact index row. Every `input_hashes[]` member has the canonical object shape `{uri, sha256, expected_identity_kind, expected_type_or_declaration}`. Every `factor_hashes` map value has that same shape, with the map key naming the factor role rather than its identity. The verifier normalizes the URI, resolves exactly one row, checks the expected kind and type/declaration, and compares the value with the indexed digest before hashing the aggregate parent. A bare digest string, URI-less member, type-less member, inferred kind, missing/duplicate row, or correctly spelled digest bound to the wrong row is invalid.

The four container identity fields—`validation-record.v1` analyzer `container_digest`, `model-spec.v1` `container_digest`, `run-manifest.v1` container-fingerprint `image_digest`, and `baseline-comparison.v1` analyzer `container_digest`—use one locator shape rather than a digest scalar:

```text
{uri, digest, expected_identity_kind=native_digest,
 expected_type_or_declaration=RAW-07_OCI_IMAGES,
 [retained_bytes={uri, digest, expected_identity_kind=raw_sha256,
                   expected_type_or_declaration=RAW-07_OCI_IMAGES}]}
```

The primary locator has exactly the four required members `{uri,digest,expected_identity_kind,expected_type_or_declaration}` before the optional nested `retained_bytes` member. Its `digest` is exactly the resolved native index-row value `<registered-native-algorithm>:<canonical-lowercase-native-digest>`. The normalized URI must resolve to exactly one `native_digest` row whose digest, kind, and `RAW-07_OCI_IMAGES` declaration all equal the locator. URI-only resolution is insufficient. The bracketed retained-byte locator is optional and independent; when present it has exactly the same four member names, its `digest` is `sha256:<64-lowercase-hex>`, and its different normalized URI resolves to one distinct `raw_sha256` row with equal digest/declaration and the same owning structured `parent_uri`. The containing structured parent explicitly carries both locators.

Both locator digest strings participate unchanged in the containing artifact's ordinary JCS projection as referenced-child identities. They are never excluded as the parent's `own_identity_field`; only the registry field already assigned to that parent may be removed. Therefore changing either child digest at a stable URI changes the direct parent artifact ID and every content-addressed ancestor. Verification never hashes an index row ID, dereferences bytes while hashing the parent, or accepts a second digest field name. The fixed literal member name `digest`, already defined by the index grammar, is reused as an exact index-value copy; repeated occurrences do not add an outer semantic identity name, so the exhaustive ledger remains exactly 77. Fields named `row_id`, `native_digest`, `retained_digest`, or another identity alias are forbidden; `native_digest` remains only the registered `identity_kind` token.

The verifier rejects a scalar or composite digest; missing, blank, `null`, all-zero, `TBD`, `GATED_*`, or other sentinel locator member; malformed, uppercase, or wrong-length digest; URI/digest mismatch; wrong kind or declaration; missing/dangling row; noncanonical or aliased URI; duplicate normalized URI; a shared native/raw row or physical file; different owning parents; and an undeclared retained copy. Omission of the entire optional `retained_bytes` member is valid and makes no retained-evidence or retention claim.

The canonical local child-sensitivity fixture hashes its ordinary parent JCS with the global §3 rule. The fixture is synthetic and normative only for content-addressing validation; it is non-normative for the actual model revision, container selection, deployment, security, retention, or research configuration. Its parent type is `model-spec.v1`, whose registered `own_identity_field=none`, so the verifier hashes `sha256(JCS(complete_resolved_fixture_object))` and no `model_spec_id` appears in the object. Each complete object contains only `schema`, `revision`, and `container_digest`; the locator contains exactly its four primary members plus the four-member nested `retained_bytes` locator.

For each of the next three code blocks, the single line is simultaneously the complete concrete JSON object and the exact canonical JCS UTF-8 byte sequence. The hashed bytes are exactly the characters between the fences: no BOM, leading or trailing whitespace, line break, or trailing LF is included. There are no ellipses, repetition shorthand, placeholders, implicit defaults, or omitted fields.

Base object and canonical bytes:

```json
{"container_digest":{"digest":"oci:sha256:1111111111111111111111111111111111111111111111111111111111111111","expected_identity_kind":"native_digest","expected_type_or_declaration":"RAW-07_OCI_IMAGES","retained_bytes":{"digest":"sha256:3333333333333333333333333333333333333333333333333333333333333333","expected_identity_kind":"raw_sha256","expected_type_or_declaration":"RAW-07_OCI_IMAGES","uri":"images/model-runtime.manifest"},"uri":"images/model-runtime.oci.ref"},"revision":"ad44","schema":"model-spec.v1"}
```

Native-only mutation object and canonical bytes:

```json
{"container_digest":{"digest":"oci:sha256:2222222222222222222222222222222222222222222222222222222222222222","expected_identity_kind":"native_digest","expected_type_or_declaration":"RAW-07_OCI_IMAGES","retained_bytes":{"digest":"sha256:3333333333333333333333333333333333333333333333333333333333333333","expected_identity_kind":"raw_sha256","expected_type_or_declaration":"RAW-07_OCI_IMAGES","uri":"images/model-runtime.manifest"},"uri":"images/model-runtime.oci.ref"},"revision":"ad44","schema":"model-spec.v1"}
```

Retained-only mutation object and canonical bytes:

```json
{"container_digest":{"digest":"oci:sha256:1111111111111111111111111111111111111111111111111111111111111111","expected_identity_kind":"native_digest","expected_type_or_declaration":"RAW-07_OCI_IMAGES","retained_bytes":{"digest":"sha256:4444444444444444444444444444444444444444444444444444444444444444","expected_identity_kind":"raw_sha256","expected_type_or_declaration":"RAW-07_OCI_IMAGES","uri":"images/model-runtime.manifest"},"uri":"images/model-runtime.oci.ref"},"revision":"ad44","schema":"model-spec.v1"}
```

The native mutation differs from base at exactly one value: the 64 native digest digits change from literal `1` digits to literal `2` digits. The retained mutation differs from base at exactly one value: the 64 retained digest digits change from literal `3` digits to literal `4` digits. Keys, URIs, kinds, declarations, nesting, and every other value remain identical.

This is the exact noncanonical one-line UTF-8 JSON input for the unchanged base. Its top-level member order is `schema`, `revision`, `container_digest`; its native-locator order is `uri`, `retained_bytes`, `expected_type_or_declaration`, `expected_identity_kind`, `digest`; and its retained-locator order is `uri`, `expected_type_or_declaration`, `expected_identity_kind`, `digest`:

```json
{"schema":"model-spec.v1","revision":"ad44","container_digest":{"uri":"images/model-runtime.oci.ref","retained_bytes":{"uri":"images/model-runtime.manifest","expected_type_or_declaration":"RAW-07_OCI_IMAGES","expected_identity_kind":"raw_sha256","digest":"sha256:3333333333333333333333333333333333333333333333333333333333333333"},"expected_type_or_declaration":"RAW-07_OCI_IMAGES","expected_identity_kind":"native_digest","digest":"oci:sha256:1111111111111111111111111111111111111111111111111111111111111111"}}
```

The verifier parses this input and JCS-canonicalizes the resulting JSON value before hashing. Directly hashing the displayed noncanonical source serialization is invalid; after JCS it must reproduce the canonical base bytes and base ID. The verifier rejects an omitted or extra member, any alternate value, `model_spec_id`, ellipsis or repetition shorthand, placeholder/default substitution, hashing before JCS, or a hashed-byte sequence containing a BOM, whitespace, or trailing LF.

These expected lowercase SHA-256 vectors are normative for this synthetic fixture and are exercised at all four container-locator field paths:

| Fixture | Only change from base | Expected parent artifact ID |
|---|---|---|
| base | none | `4ed18e126cd11687b25abbae48a7492536a4cd42b65e7cff6551397db976df3b` |
| native mutation | replace only the valid native locator `digest` at the same URI | `ea12f2fda667d563431de2147c01957dc639260901923d31cc9e1b1eaa251d33` |
| retained mutation | replace only the valid retained-byte locator `digest` at the same URI | `784704fb8c6a5e2f18aec453ccb46d7dd5f3e75a3d9536b7d8c7881408a171f0` |
| order permutation | serialize the unchanged base members in a different member order before JCS | unchanged: `4ed18e126cd11687b25abbae48a7492536a4cd42b65e7cff6551397db976df3b` |

These native digests are reproducibility identifiers, not authentication, signatures, provenance trust, or authorization. Including or omitting an optional retained-byte locator does not choose retention duration, deletion behavior, storage location, encryption, access control, or any security policy; those project-owner gates remain unresolved.

### 11.2 Internally consistent resolved YAML example

The single example below is a human-authored bundle whose resolver validates and content-addresses each named object. `N/V/D` comments identify normative, verified, or derived fields; every concrete `P` value is a proposed, nonbinding example; every `G` value requires project-owner or feasibility approval. No `P` value below—including hardware, attention, geography, limits, security, or workload choices—is a selected project default. `GATED_*` tokens are authoring notation only and are rejected in emitted resolved artifacts. The resolver first replaces every gate with an approved concrete value, resolves and hashes children, inserts those child IDs into parents, computes each artifact ID using the registry, and writes the ID only to the out-of-band artifact index or the one registered own field. The generic objects shown here register `none`, so they contain no circular own-ID placeholder.

```yaml
schema: experiment-bundle.v1
experiment_name: qwen-live-e2e-placement-policy-control
vary_only: placement_policy
content_addressing:
  rule: SHA256_OF_JCS_WITH_REGISTERED_OWN_IDENTITY_FIELD_REMOVED
  artifact_identity_registry_sha256: GATED_REGISTRY_SHA256_BOUND_BY_SCHEMA_BUNDLE
  generic_factor_identity_location: OUT_OF_BAND_ARTIFACT_INDEX
  placeholders_allowed_in_emitted_artifacts: false
  closure_counts: {immediate_structured: 57, conditional_gated: 2, raw_native: 20,
                   runtime_only_excluded: 5, audited_identity_names: 77}
  unclassified_identity_behavior: FAIL_CLOSED

model:
  schema: model-spec.v1
  model_id_source: Qwen/Qwen3-30B-A3B
  revision: ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
  safetensors_index_sha256: GATED_COMPUTE_FROM_PINNED_INDEX
  tokenizer_revision: ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
  chat_template_revision: ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
  generation_config_revision: ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
  transformers_version: 4.51.3
  pytorch_version: GATED
  accelerator_backend: GATED  # P, nonbinding; one of CUDA | MPS | CPU
  accelerator_version: GATED  # CUDA toolkit/driver version, macOS/MPS version, or NA for CPU
  driver_version: GATED  # applicable to CUDA hosts only
  container_digest:
    uri: images/model-runtime.oci.ref
    digest: GATED_RESOLVER_NATIVE_DIGEST
    expected_identity_kind: native_digest
    expected_type_or_declaration: RAW-07_OCI_IMAGES
  attention_backend: eager  # P, nonbinding; exact backend is G
  dtype: BF16
  quantization: NONE
  layers: 48
  hidden_dimension: 2048
  experts_per_layer: 128
  top_k: 8
  normalize_top_k: true
  shared_weight_strategy: COORDINATOR_CENTRALIZED
  max_context_tokens: 32768  # P, nonbinding; owner must approve

workload:
  schema: workload-spec.v1
  prompt_manifest_uri: artifacts/prompts/representative-v1.jsonl
  prompt_manifest_sha256: GATED
  dataset_source: GATED
  dataset_license_class: GATED
  arrival:
    kind: CLOSED_LOOP
    concurrency: 2  # P, nonbinding
  request_count: 100  # P, nonbinding
  prompt_tokens: {class: SHORT, min: 32, max: 256}  # P, nonbinding
  generation_tokens: {class: SHORT, max: 64}  # P, nonbinding
  generation: {temperature: 0.0, greedy: true, stream: true}  # P, nonbinding oracle shape
  warm_state: WARM  # P, nonbinding
  warmup: {requests: 10, stability_window_requests: 5, max_cv: 0.05}  # P, nonbinding

hardware:
  schema: hardware-inventory.v1
  profiles:
    ref-h100-80g:  # P, nonbinding hardware candidate; reference GPU is G
      cpu_model: GATED
      cpu_cores: GATED
      host_ram_bytes: GATED
      gpu_model: NVIDIA H100 80GB HBM3  # P, nonbinding candidate
      gpu_count: 1  # P, nonbinding candidate
      vram_bytes_each: 85899345920  # P, candidate inventory value to verify
      nic_nominal_mbps: 100000  # P, candidate inventory value to verify
      usable_weight_bytes: GATED_MEASURE
      declared: {source: vendor_spec, compute_tflops: null, memory_bandwidth_gbps: null}
      measured_expert_profile_id: null
    worker-h100-80g:  # P, nonbinding hardware candidate; worker GPU is G
      cpu_model: GATED
      cpu_cores: GATED
      host_ram_bytes: GATED
      gpu_model: NVIDIA H100 80GB HBM3  # P, nonbinding candidate
      gpu_count: 1  # P, nonbinding candidate
      vram_bytes_each: 85899345920  # P, candidate inventory value to verify
      nic_nominal_mbps: 10000  # P, candidate inventory value to verify
      usable_weight_bytes: 68719476736  # P, illustrative only; must be measured
      declared: {source: vendor_spec, compute_tflops: null, memory_bandwidth_gbps: null}
      measured_expert_profile_id: null
    worker-mac-studio-m2-ultra:  # P, nonbinding hardware candidate; Apple Silicon/MPS field shape
      cpu_model: GATED
      cpu_cores: GATED
      host_ram_bytes: GATED
      accelerator_backend: MPS  # P, nonbinding candidate
      accelerator_model: Apple M2 Ultra  # P, nonbinding candidate
      accelerator_count: 1  # P, nonbinding candidate
      unified_memory_bytes: GATED  # P; shared CPU/GPU pool, not a separate VRAM field
      nic_nominal_mbps: 10000  # P, candidate inventory value to verify
      usable_weight_bytes: GATED_MEASURE
      declared: {source: vendor_spec, compute_tflops: null, memory_bandwidth_gbps: null}
      measured_expert_profile_id: null

topology:
  schema: topology-spec.v1
  topology_id_hint: lab-asymmetric-01
  physical_site_link_inventory: null  # optional physical-site-link-inventory.v1; owner/deployment gated
  physical_sites:
    - {id: lab-a, kind: private_lab, country: US}  # P, nonbinding site label
    - {id: lab-b, kind: private_lab, country: US}  # P, nonbinding site label
  logical_regions: [{id: us-west}, {id: us-east}]  # P labels, not physical-WAN evidence
  lan_domains: [{id: lan-a}, {id: lan-b}]
  nodes:
    - id: coordinator-0
      role: COORDINATOR
      physical_site_id: lab-a
      logical_region_id: us-west
      lan_domain_id: lan-a
      management_address: 192.168.10.10
      data_address: 10.80.0.10
      data_interface: enp65s0f0
      hardware_profile_id: ref-h100-80g
      source_uplink_mbps: 1000
      destination_downlink_mbps: 1000
    - id: worker-1
      role: EXPERT_WORKER
      physical_site_id: lab-b
      logical_region_id: us-east
      lan_domain_id: lan-b
      management_address: 192.168.10.11
      data_address: 10.80.0.11
      data_interface: enp65s0f0
      hardware_profile_id: worker-h100-80g
      source_uplink_mbps: 500
      destination_downlink_mbps: 500
  links:
    - id: coordinator-0__to__worker-1
      source: coordinator-0
      destination: worker-1
      one_way_delay_ms: 20  # P, nonbinding impairment
      bandwidth_mbps: 100  # P, nonbinding impairment
      jitter: {distribution: normal, deviation_ms: 3, correlation_percent: 0}
      loss: {model: random, percent: 0.10, seed: 8101}
      queue_limit_packets: 10000  # P, nonbinding limit; owner gate 5
      physical_link_evidence_id: null  # no physical-WAN classification in this emulated example
    - id: worker-1__to__coordinator-0
      source: worker-1
      destination: coordinator-0
      one_way_delay_ms: 50  # P, nonbinding impairment
      bandwidth_mbps: 20  # P, nonbinding impairment
      jitter: {distribution: normal, deviation_ms: 8, correlation_percent: 0}
      loss: {model: random, percent: 0.01, seed: 8102}
      queue_limit_packets: 10000  # P, nonbinding limit; owner gate 5
      physical_link_evidence_id: null  # no physical-WAN classification in this emulated example

latency_discovery:
  schema: latency-discovery-snapshot.v1
  discovery_spec: {schema: latency-discovery-spec.v1, uri: artifacts/placement/latency-discovery-spec.json,
                   artifact_id: GATED_RESOLVER_SHA256}
  source_kind: ACTIVE_MEASURED
  network_condition: EMULATED_PROFILE
  topology_spec_sha256: GATED_RESOLVER_SHA256
  topology_proof_sha256: GATED_AFTER_TOPOLOGY_PROOF
  candidate_slot_inventory_sha256: GATED_RESOLVER_SHA256
  measurement_window: {start_utc: GATED, end_utc: GATED}
  required_cell_count: GATED_OWNER_STUDY_CHOICE
  valid_cell_count: GATED_AFTER_DISCOVERY
  coverage: GATED_AFTER_DISCOVERY
  age_at_admission_ns: GATED_AFTER_DISCOVERY
  cell_key: [source_coordinator_id, candidate_slot_id, endpoint_id, transport, TLS_mode,
             request_payload_class, response_payload_class, request_bytes, response_bytes, concurrency_class,
             network_condition, uncertainty_coverage_cell_id]
  observation: COORDINATOR_MONOTONIC_ROUND_TRIP_NS
  one_way_inference: FORBIDDEN
  interpolation_domain_schema: interpolation-domain.v1
  interpolation_domain_sha256: GATED_OWNER_PREREGISTRATION
  admission_policy: {schema: measurement-admission-policy.v1,
                     artifact_id: GATED_OWNER_PREREGISTRATION}
  cells_uri: artifacts/placement/discovery-cells.parquet
  cells_sha256: GATED_AFTER_DISCOVERY

routing_partition:
  schema: routing-partition-manifest.v1
  grouping_unit: SOURCE_DOCUMENT_CONVERSATION_OR_PROMPT_TEMPLATE
  assignment: SALTED_STABLE_HASH_WITHIN_WORKLOAD_STRATUM
  salt_commitment_sha256: GATED_BEFORE_CAPTURE
  train_trace_sha256: GATED_AFTER_CORRECTNESS_QUALIFIED_CAPTURE
  validation_trace_sha256: GATED_AFTER_CORRECTNESS_QUALIFIED_CAPTURE
  test_trace_commitment_sha256: GATED_SEALED
  leakage_audit: GATED_ANALYZER_RESULT

activation_statistics:
  schema: activation-statistics.v1
  source_partition_sha256: GATED_RESOLVER_SHA256
  permitted_splits: [TRAIN, VALIDATION]
  strata: [PREFILL, DECODE]
  estimators: [SELECTION_FREQUENCY, ROUTING_WEIGHT_MASS, SAME_LAYER_COACTIVATION, CROSS_LAYER_DESCRIPTIVE]
  record_fields: [statistic_kind, split, token_phase, workload_stratum, coordinates,
                  numerator, denominator, eligible_count, excluded_count, value_status, value, na_reason]
  same_layer_domain: UNORDERED_E_LT_E_PRIME_NO_DIAGONAL
  cross_layer_domain: ORDERED_POSITIVE_CONFIGURED_OFFSETS
  cross_layer_offsets: GATED_OWNER_STUDY_CHOICE
  zero_denominator: TYPED_NA_WITH_COUNTS
  na_to_zero: FORBIDDEN
  data_uri: artifacts/placement/activation-statistics.parquet
  data_sha256: GATED_ANALYZER_SHA256

constraint_set:
  schema: constraint-set.v1
  candidate_slots:
    schema: candidate-slot-inventory.v1
    uri: artifacts/placement/candidate-slots.json
    sha256: GATED_RESOLVER_SHA256
    expected_identity_kind: structured_jcs
    expected_type_or_declaration: candidate-slot-inventory.v1
    identity_fields: [worker_id, tier, slot_ordinal]
    candidate_slot_id_rule: SHA256_JCS_IDENTITY_FIELDS
    policy_mutation: FORBIDDEN
  worker_tier_limits:
    schema: worker-tier-limits.v1
    uri: artifacts/placement/worker-tier-limits.json
    sha256: GATED_RESOLVER_SHA256
    expected_identity_kind: structured_jcs
    expected_type_or_declaration: worker-tier-limits.v1
    fields: [worker_id, tier, allocatable_bytes, reserve_bytes, usable_bytes, slot_limit, tier_rank]
  expert_contracts:
    schema: expert-contracts.v1
    uri: artifacts/placement/expert-contracts.json
    sha256: GATED_RESOLVER_SHA256
    expected_identity_kind: structured_jcs
    expected_type_or_declaration: expert-contracts.v1
    fields: [layer_id, expert_id, exact_bytes, dtype, tensor_digest, model_revision, runtime_compatibility]
  compatibility_matrix: {schema: expert-slot-compatibility-matrix.v1,
                         uri: artifacts/placement/compatibility-matrix.json,
                         compatibility_matrix_sha256: GATED_RESOLVER_SHA256}
  failure_domains: {schema: failure-domain-inventory.v1,
                    uri: artifacts/placement/failure-domains.json,
                    failure_domains_sha256: GATED_RESOLVER_SHA256}
  replica_bounds_and_caps: {schema: replica-bounds-and-caps.v1,
                            uri: artifacts/placement/replica-bounds-and-caps.json,
                            replica_bounds_and_caps_sha256: GATED_RESOLVER_SHA256}
  resilience_failure_sets: {schema: resilience-failure-sets.v1,
                            uri: artifacts/placement/resilience-failure-sets.json,
                            resilience_failure_sets_sha256: GATED_RESOLVER_SHA256}
  comparison_budgets:
    phase_2a: {exact_replica_count: 6144, exact_resident_bytes: 57982058496}
    phase_2b: GATED_OWNER_EXACT_COMMON_REPLICA_AND_BYTE_BUDGET
  transition_headroom: {static_phase_2a_2b: NOT_APPLICABLE, dynamic_phase_4: GATED_OWNER_STUDY_CHOICE}

objective_spec:
  schema: objective-spec.v1
  formula_version: JOINT_RPC_PLUS_SERVICE_V1
  grouped_route_source_sha256: GATED_CORRECTNESS_QUALIFIED_TRAIN_OR_ALLOWED_VALIDATION_TRACE
  constraint_set_sha256: GATED_RESOLVER_SHA256
  replica_bytes_source: CONSTRAINT_SET_EXPERT_SIZE_TABLE
  movement: {mode: IMMUTABLE_STARTUP_ZERO, prior_plan_sha256: null}
  latency_surface:
    snapshot_sha256: GATED_AFTER_ACTIVE_DISCOVERY
    cells_sha256: GATED_AFTER_DISCOVERY
    key: [source_coordinator_id, candidate_slot_id, endpoint_id, transport, TLS_mode,
          request_payload_class, response_payload_class, request_bytes, response_bytes, concurrency_class,
          network_condition, latency_discovery_snapshot_sha256, uncertainty_coverage_cell_id]
    use_count_per_required_slot_branch: 1
  service_surface:
    calibration_set_sha256: GATED_AFTER_CALIBRATION
    curves_sha256: GATED_COMPUTE_AFTER_CALIBRATION
    key: [candidate_slot_id, worker_id, tier, hardware_revision, software_revision,
          model_revision, expert_class, dtype, rows, payload_class, concurrency_class,
          residency_state, service_calibration_snapshot_sha256]
  coordinator_fixed_packing_staging_cost: {mode: EXPLICIT_ZERO, value_ns: 0}
  freshness_and_coverage_policy_sha256: GATED_OWNER_PREREGISTRATION
  missing_stale_undercovered_or_out_of_domain: STRUCTURED_INFEASIBILITY
  extrapolation: FORBIDDEN

placement_study_snapshot:
  schema: placement-study-snapshot.v1
  bound_inputs: [model, hardware, topology, latency_discovery, calibration, constraint_set,
                 objective_spec, scheduler, routing_partition, activation_statistics]
  constraint_set_sha256: GATED_RESOLVER_SHA256
  objective_spec_sha256: GATED_RESOLVER_SHA256
  scheduler_spec_sha256: GATED_RESOLVER_SHA256
  feature_access: {train: ALLOW, validation: PREREGISTERED_ONLY, test: COMMITMENT_ONLY_UNTIL_UNSEAL}
  test_seal_commitment_sha256: GATED_BEFORE_PLANNING
  online_updates: FORBIDDEN

placement_policy:
  schema: placement-policy-spec.v1
  policy: ManualExplicit
  source_sha256: GATED_POLICY_SOURCE
  feature_allowlist: [MODEL, HARDWARE, TOPOLOGY, CAPACITY]
  solver: {name: explicit-validator, version: GATED, seed: 74291,
           limits_schema: solver-limits.v1, limits_sha256: GATED}
  objective_spec_sha256: GATED_RESOLVER_SHA256
  constraint_set_sha256: GATED_RESOLVER_SHA256
  surrogate_scheduler_sha256: GATED_EQUAL_SCHEDULER_SPEC_SHA256
  quantization: {time_unit: ns, byte_unit: byte}
  canonical_tie_break: [layer_id, expert_id, worker_id, tier_rank, slot_ordinal]
  online_updates: FORBIDDEN

placement:
  schema: placement-plan.v2
  placement_epoch: 1
  placement_study_snapshot_sha256: GATED_RESOLVER_SHA256
  placement_policy_spec_sha256: GATED_RESOLVER_SHA256
  constraint_set_sha256: GATED_RESOLVER_SHA256
  objective_spec_sha256: GATED_RESOLVER_SHA256
  scheduler_spec_sha256: GATED_RESOLVER_SHA256
  policy: ManualExplicit
  coverage_rule: EVERY_LAYER_EXPERT_EXACTLY_ONE
  exact_replica_count_budget: 6144
  exact_resident_byte_budget: 57982058496
  assignment_template:
    description: resolve all 6144 layer/expert pairs before execution
    layer_range_inclusive: [0, 47]
    expert_range_inclusive: [0, 127]
    worker_cycle: [worker-1]
    candidate_slot_rule: NEXT_CANONICAL_FREE_SLOT
    digest_source: pinned_safetensors_index
  assignments_override:
    - {layer_id: 12, expert_id: 37, worker_id: worker-1,
       tier: DEVICE, slot_ordinal: GATED_RESOLVED_SLOT_ORDINAL,
       candidate_slot_id: GATED_SHA256_JCS_WORKER_TIER_SLOT,
       replica_id: GATED_SHA256_JCS_MODEL_EXPERT_CANDIDATE_SLOT,
       expert_digest: GATED_COMPUTE_FROM_PINNED_WEIGHTS}
  objective_components: {p95_predicted_remote_ns: GATED, mean_predicted_remote_ns: GATED,
                         mean_worker_fanout: GATED, replica_bytes: 57982058496, movement_bytes: 0}
  feasibility_proof_sha256: GATED_AFTER_COMPLETE_CONSTRAINT_CHECK
  canonical_tie_break: [layer_id, expert_id, worker_id, tier_rank, slot_ordinal]

placement_application:
  schema: placement-application-record.v1
  placement_plan_sha256: GATED_RESOLVER_SHA256
  state_sequence: [RESOLVED, PREPARED, LOADING, VERIFIED, COMMITTED, ADMISSION_OPEN]
  verification: {coverage: GATED, digests: GATED, capacity: GATED, residency: GATED, incarnation: GATED}
  directory_snapshot_sha256: GATED_AFTER_ATOMIC_PUBLICATION

directory_snapshot:
  schema: directory-snapshot.v1
  directory_revision: 1
  placement_epoch: 1
  snapshot_time_utc: 2026-08-19T00:00:00Z  # P example; resolver records actual time
  immutable_for_run: true
  worker_incarnations: {worker-1: 018f-example-worker-incarnation}  # P example identity
  replica_records_uri: artifacts/directory/directory-snapshot-1.replicas.json
  replica_records_schema: directory-replica-records.v1
  replica_records_sha256: GATED_COMPUTE_AFTER_STARTUP_LOAD
  replica_record_count: 6144
  required_state: DEVICE_WARM  # P, nonbinding startup policy
  health: SERVING

scheduler:
  schema: scheduler-spec.v1
  policy: FixedPrimary
  policy_version: 1
  candidate_order_version: canonical-slot-order.v1
  candidate_order_key: [worker_id_utf8_bytes, tier_rank_from_constraint_set, slot_ordinal_uint, replica_id_bytes]
  candidate_order_encoding: {strings: UTF8_BYTEWISE_CASE_SENSITIVE, integers: UNSIGNED_BASE10_VALUE, digests: RAW_SHA256_BYTES}
  tie_rule: LEXICOGRAPHIC_MIN
  ignores: [PLAN_ROW_ORDER, SERIALIZATION_ORDER, DIRECTORY_INSERTION_ORDER, POLICY_OUTPUT_ORDER]
  eligibility_state: DEVICE_WARM
  max_snapshot_age_us: 5000  # P, nonbinding; owner gate 5
  max_attempts_per_logical_batch: 2  # P, nonbinding; owner gate 5
  automatic_grpc_retry: false
  hedging: false
  no_replica_behavior: FAIL_REQUEST
  seed_domain: scheduler

batching:
  schema: batching-spec.v1
  cohort:
    max_requests: 2  # P, nonbinding; owner gate 5
    admission_window_us: 500  # P, nonbinding; owner gate 5
    finished_row_policy: INERT_UNTIL_COHORT_END
  dispatch:
    packing_key: [target_worker_id, layer_id]
    max_rows_per_expert_segment: 4096  # P, nonbinding; owner gate 5
    max_rpc_payload_bytes: 33554432  # P, nonbinding; owner gate 5
    grpc_max_message_bytes: 41943040  # P, nonbinding; owner gate 5
    deterministic_chunking: true
  worker_microbatch:
    max_rows: 4096  # P, nonbinding; owner gate 5
    max_bytes: 33554432  # P, nonbinding; owner gate 5
    max_wait_us: 100  # P, nonbinding; owner gate 5
    max_concurrent_kernels: 1  # P, nonbinding; owner gate 5

faults:
  schema: fault-plan.v1
  events:
    - id: warmup-noop-marker
      at_run_offset_ms: 0
      duration_ms: 0
      target: {kind: CONTROLLER, id: topology-controller}
      action: RECORD_ONLY
      mechanism: none
      seed: 9001
  disabled_catalog:
    - {id: partition-worker-1, target: {kind: DIRECTIONAL_LINK, id: coordinator-0__to__worker-1}, action: BLACKHOLE}
    - {id: lose-expert, target: {kind: EXPERT_REPLICA, id: l12-e37-r1}, action: UNLOAD}
    - {id: kill-worker, target: {kind: WORKER, id: worker-1}, action: TERMINATE_PROCESS}

execution:
  schema: execution-mode-spec.v1
  execution_mode: LIVE_E2E
  fidelity:
    model_path: FULL
    routing: ORIGINAL_LIVE
    activations: LIVE_CAUSAL
    compute: REAL
    network: EMULATED
    timing: WALL_CLOCK_MEASURED
  claim_allowlist_version: claims.v1

calibration:
  schema: calibration-set.v1
  calibration_id_hint: pre-run-calibration-01  # P example
  environment_fingerprint_schema: environment-fingerprint.v1
  environment_fingerprint_sha256: GATED_COMPUTE_FROM_RESOLVED_ENVIRONMENT
  hardware_inventory_sha256: GATED_COMPUTE_FROM_RESOLVED_HARDWARE
  latency_discovery_snapshot_sha256: GATED_AFTER_ACTIVE_DISCOVERY
  latency_authority: ACTIVE_MEASURED
  configured_emulation_sha256: GATED_RESOLVER_SHA256
  expert_service_curves:
    snapshot_schema: service-calibration-snapshot.v1
    uri: artifacts/calibration/expert-service-curves.parquet
    sha256: GATED_COMPUTE_AFTER_CALIBRATION
    expected_identity_kind: raw_sha256
    expected_type_or_declaration: RAW-13_SERVICE_CURVES_PARQUET
    service_calibration_snapshot_sha256: GATED_COMPUTE_AFTER_CALIBRATION
    keys: [candidate_slot_id, worker_id, tier, hardware_revision, software_revision,
           model_revision, expert_class, dtype, rows, payload_class, concurrency_class,
           residency_state, service_calibration_snapshot_sha256]
    interpolation_domain_sha256: GATED_OWNER_PREREGISTRATION
    transport_included: false
  topology_probes:
    uri: artifacts/calibration/topology-probes.parquet
    sha256: GATED_COMPUTE_AFTER_CALIBRATION
    expected_identity_kind: raw_sha256
    expected_type_or_declaration: RAW-14_TOPOLOGY_PROBES_PARQUET
    ordered_link_count: 2

seeds:
  master: 74291
  derivation: "SHA256(master || domain || repetition_index)[:64bits]"
  domains: [workload, sampling, scheduler, placement, network_emulation, faults, replay_service]

run_manifest:
  schema: run-manifest.v1
  experiment_id: experiment-qwen-moe-001  # P example
  trial_id: trial-placement-001  # P example
  run_id: 018f-example-run-id  # P example; actual run uses UUIDv7
  manifest_created_at_utc: 2026-08-19T00:00:00Z  # P example; resolver records actual time
  schema_bundle_sha256: GATED_RESOLVER_SHA256
  artifact_identity_registry_sha256: GATED_REGISTRY_SHA256_BOUND_BY_SCHEMA_BUNDLE
  artifact_identity_rule: SHA256_OF_JCS_WITH_REGISTERED_OWN_IDENTITY_FIELD_REMOVED
  all_referenced_artifact_ids_recomputed: true
  api_compatibility_manifest_schema: api-compatibility-manifest.v1
  api_compatibility_manifest_sha256: GATED_OWNER_DECISION
  factor_hashes:
    model: {uri: factors/model.json, sha256: GATED_RESOLVER_SHA256,
            expected_identity_kind: structured_jcs, expected_type_or_declaration: model-spec.v1}
    workload: {uri: factors/workload.json, sha256: GATED_RESOLVER_SHA256,
               expected_identity_kind: structured_jcs, expected_type_or_declaration: workload-spec.v1}
    hardware: {uri: factors/hardware.json, sha256: GATED_RESOLVER_SHA256,
               expected_identity_kind: structured_jcs, expected_type_or_declaration: hardware-inventory.v1}
    topology: {uri: factors/topology.json, sha256: GATED_RESOLVER_SHA256,
               expected_identity_kind: structured_jcs, expected_type_or_declaration: topology-spec.v1}
    placement: {uri: artifacts/placement/plan.json, sha256: GATED_RESOLVER_SHA256,
                expected_identity_kind: structured_jcs, expected_type_or_declaration: placement-plan.v2}
    scheduler: {uri: factors/scheduler.json, sha256: GATED_RESOLVER_SHA256,
                expected_identity_kind: structured_jcs, expected_type_or_declaration: scheduler-spec.v1}
    batching: {uri: factors/batching.json, sha256: GATED_RESOLVER_SHA256,
               expected_identity_kind: structured_jcs, expected_type_or_declaration: batching-spec.v1}
    execution_mode: {uri: factors/mode.json, sha256: GATED_RESOLVER_SHA256,
                     expected_identity_kind: structured_jcs, expected_type_or_declaration: execution-mode-spec.v1}
    fault_plan: {uri: factors/faults.json, sha256: GATED_RESOLVER_SHA256,
                 expected_identity_kind: structured_jcs, expected_type_or_declaration: fault-plan.v1}
    calibration: {uri: artifacts/calibration/calibration-set.json, sha256: GATED_RESOLVER_SHA256,
                  expected_identity_kind: structured_jcs, expected_type_or_declaration: calibration-set.v1}
    placement_study_snapshot: {uri: artifacts/placement/study-snapshot.json, sha256: GATED_RESOLVER_SHA256,
                               expected_identity_kind: structured_jcs, expected_type_or_declaration: placement-study-snapshot.v1}
    placement_policy_spec: {uri: factors/placement-policy.json, sha256: GATED_RESOLVER_SHA256,
                            expected_identity_kind: structured_jcs, expected_type_or_declaration: placement-policy-spec.v1}
    constraint_set: {uri: artifacts/placement/constraint-set.json, sha256: GATED_RESOLVER_SHA256,
                     expected_identity_kind: structured_jcs, expected_type_or_declaration: constraint-set.v1}
    objective_spec: {uri: artifacts/placement/objective-spec.json, sha256: GATED_RESOLVER_SHA256,
                     expected_identity_kind: structured_jcs, expected_type_or_declaration: objective-spec.v1}
  placement_contracts:
    routing_partition_manifest_sha256: GATED_RESOLVER_SHA256
    activation_statistics_sha256: GATED_ANALYZER_SHA256
    constraint_set_sha256: GATED_RESOLVER_SHA256
    objective_spec_sha256: GATED_RESOLVER_SHA256
    scheduler_spec_sha256: GATED_RESOLVER_SHA256
    feasibility_proof_sha256: GATED_ANALYZER_SHA256
    comparison_contract_sha256: GATED_BEFORE_TEST_UNSEAL
    start_state_record_sha256: GATED_BEFORE_ADMISSION
    placement_application_record_sha256: GATED_AFTER_ADMISSION_OPEN
    test_unseal_event_id: GATED_AFTER_ALL_PLAN_AND_COMPARATOR_HASHES
  evidence_children:
    raw_events: {schema: raw-event-manifest.v1, artifact_id: GATED_AFTER_RUN}
    attempt_status: {schema: attempt-status.v1, artifact_id: GATED_AFTER_RUN}
    metric_summary: {schema: metric-summary.v1, artifact_id: GATED_AFTER_ANALYSIS}
  cell: {cell_id: GATED_STUDY_MATRIX, pair_block_id: GATED, repetition_index: GATED, order_index: GATED}
  initial_directory_snapshot: {uri: artifacts/directory/directory-snapshot-1.json,
                               sha256: GATED_RESOLVER_SHA256,
                               expected_identity_kind: structured_jcs,
                               expected_type_or_declaration: directory-snapshot.v1}
  vary_only: placement_policy
  seed_derivation: {master: 74291, algorithm: "SHA256(master || domain || repetition_index)[:64bits]"}
  warm_state: WARM  # P, nonbinding
  software_fingerprint: {transformers: 4.51.3, pytorch: GATED, accelerator_backend: GATED, accelerator_version: GATED, driver: GATED}
  container_fingerprint:
    image_digest: {uri: images/run-runtime.oci.ref,
                   digest: GATED_RESOLVER_NATIVE_DIGEST,
                   expected_identity_kind: native_digest,
                   expected_type_or_declaration: RAW-07_OCI_IMAGES}
  trace_manifest: {schema: trace-manifest.v1, uri: traces/trace_manifest.json,
                   sha256: GATED_AFTER_RUN,
                   expected_identity_kind: structured_jcs,
                   expected_type_or_declaration: trace-manifest.v1}
  output_uri: results/experiments/EXPERIMENT/trials/TRIAL/attempts/RUN/
  event_format: parquet
  router_score_capture: ALL_REAL_TOKENS_ALL_LAYERS  # P, subject to telemetry gate 8
  activation_capture: NONE  # P, nonbinding
  immutable_attempts: true
  publish_marker: COMPLETE
  retention_policy_id: GATED_OWNER_DECISION
  security_policy_id: GATED_OWNER_DECISION
  retention_policy_schema: retention-policy.v1
  security_policy_schema: security-policy.v1
```

The candidate-slot artifact is expanded and hashed before any policy runs; the `assignment_template` then expands into 6,144 concrete `(expert,candidate_slot)` assignments before plan validation and hashing. The override must be identical to its generated row or replace it deterministically, and its replica ID is derived from the model/expert/candidate-slot identity rather than its row position. The example is only arithmetically feasible because its illustrative validated worker budget (64 GiB) exceeds the derived 54 GiB expert residency; executable feasibility still requires the content-addressed per-slot and per-worker/tier proof plus every objective/discovery admission predicate. If any fails, validation blocks execution rather than silently changing the candidate universe, offloading, quantizing, extrapolating or imputing latency/service costs, treating `NA` as zero, omitting experts, or substituting a smaller model. The example's null physical-link evidence makes every physical-WAN aggregate `NA` even though its logical regions differ.

## 12. Telemetry schema and integrity

### 12.1 Common event envelope

```text
schema_name, schema_version
event_id, logical_event_id, event_type, event_seq
experiment_id, trial_id, run_id
trace_id, span_id, parent_span_id
process_id, process_start_id, boot_id
node_id, worker_id, clock_domain_id
ts_wall_utc_ns, ts_mono_ns
clock_offset_estimate_ns, clock_uncertainty_ns
phase=WARMUP|MEASURE
session_id, request_id, token_phase, token_index, absolute_token_position
layer_id, route_id, topk_slot, expert_id
logical_expert_batch_id, worker_layer_dispatch_id, attempt_id, attempt_round, worker_batch_id
replica_id, placement_epoch, topology_epoch, directory_revision, worker_incarnation
study_snapshot_id, placement_policy_spec_id, cell_id, pair_block_id, repetition_index
status, error_code, attributes
```

Required typed events are:

```text
run_start, run_end
request_send_start, request_submitted, cohort_wait_start, cohort_wait_end
request_admitted, request_rejected
prefill_start, prefill_end, decode_step_start, decode_step_end
shared_compute_start, shared_compute_end
router_start, route_decided, router_end
dispatch_pack_start, logical_expert_batch_sealed, dispatch_pack_end, dispatch_ready
scheduler_decision_start, scheduler_decision_end
coordinator_request_stage_start, coordinator_request_stage_end
coordinator_request_serialize_start, coordinator_request_serialize_end
rpc_enqueue, rpc_write_complete, rpc_received
worker_queue_enter, worker_queue_start
host_to_device_start, host_to_device_end
worker_compute_start, worker_compute_end
device_to_host_start, device_to_host_end
response_write, coordinator_response_received
coordinator_response_stage_start, coordinator_response_stage_end
coordinator_response_deserialize_start, coordinator_response_deserialize_end
response_validation_start, response_validation_end
expert_result_accepted, expert_result_discarded
combine_start, combine_end
logits_start, logits_end, sampling_start, sampling_end, token_committed
client_stream_write_start, client_stream_write_end
client_token_received, client_final_response, request_complete
expert_load_start, expert_load_end, residency_transition
directory_update, health_update
topology_apply_start, topology_apply_end, topology_applied, topology_probe
latency_probe_start, latency_probe_end, latency_probe_result, discovery_snapshot_sealed
routing_partition_sealed, test_partition_unsealed, activation_statistics_produced
constraint_set_sealed, objective_lookup, objective_tuple_recomputed
placement_plan_start, placement_plan_iteration, placement_plan_end
placement_feasibility_checked, placement_apply_start, placement_apply_state, placement_apply_end
placement_ready, placement_commit, placement_abort
fault_scheduled, fault_apply_start, fault_apply_end, fault_observed
resource_sample, tc_counter_sample, telemetry_drop
```

Phase-4-only event names are reserved but are `NA` and absent from Phase 1/2 evidence: `placement_rollout_prepare`, `replica_add`, `placement_rollout_verify`, `placement_rollout_cas_commit`, `replica_drain`, `replica_remove`, `placement_rollback`, and `placement_rollout_finalize`. Merely emitting or documenting these names cannot support a rollout claim.

Every start/end pair shares one `span_id`, has a declared parent, and is complete or the claim-specific completeness predicate fails. Critical-path categories are derived only from these typed pairs:

| Critical-path category | Canonical typed span or edge |
|---|---|
| admission/cohort wait | `request_submitted -> request_admitted` plus `cohort_wait_start/end` |
| shared prefill/decode compute | `shared_compute_start/end` (attention, normalization, other non-MoE shared work) |
| routing | `router_start/end`; `route_decided` is the contained decision event |
| dispatch packing | `dispatch_pack_start/end` |
| scheduler | `scheduler_decision_start/end` |
| coordinator serialization/staging | request stage/serialize and response stage/deserialize spans |
| RPC residual | coordinator `rpc_enqueue -> coordinator_response_received` minus complete worker-local residence spans |
| worker queue/cold load/expert device compute | existing queue, load, copy, and compute pairs |
| response validation | `response_validation_start/end` |
| combination | `combine_start/end` |
| logits/sampling | distinct `logits_start/end` and `sampling_start/end` |
| client streaming | `client_stream_write_start/end`; client receipt remains endpoint evidence, not server CPU time |

`prefill_start/end` and `decode_step_start/end` are enclosing lifecycle spans, not substitutes for these exclusive categories. Overlap is resolved from the request/token DAG; an interval assigned to one exclusive category cannot be charged to another.

`route_decided` stores all 128 router scores in original score dtype, exact selected IDs and normalized weights, top-k, padding/activity status, any capacity/drop decision (expected none), and router implementation revision. RPC/batch records store tensor shape/dtype/order, logical activation/response bytes, serialized payload bytes, measured wire bytes when available, chunk/segment/row counts, compression, retry/hedge status, both attempt and accepted-result locality, endpoint physical-site/logical-region IDs, ordered topology-link ID, and any validated physical-link-evidence ID. Discovery events bind the complete `L_rpc` cell key, candidate-slot inventory hash, payload/concurrency classes, topology/proof hashes, coordinator-monotonic round-trip sample, outcome, and probe overhead. `activation_statistics_produced` binds the full-domain row-set hash and counts of `VALUE` and each typed `NA` by split/phase/stratum/statistic. `constraint_set_sealed` binds every leaf-table hash and root hash. Each `objective_lookup` binds objective/calibration/discovery hashes, term kind, complete lookup key, cell/row ID, interpolation endpoints, returned integer value, and rejection reason; `objective_tuple_recomputed` binds the ordered lookup-event set and derived tuple. Planning events additionally bind feature-access audit, common constraint hash, constraint counts/slacks, candidate iteration, solver state, stopping reason, and any valid certificate/bound. Startup events bind every state transition, assignment digest/residency/incarnation check, atomic directory publication, or pre-commit abort.

### 12.2 Clock and timestamp rules

- Durations use monotonic timestamps from the same `clock_domain_id` only; raw monotonic timestamps from different hosts are never subtracted.
- Wall time is correlation metadata, not a duration clock.
- Coordinator-observed RPC duration and worker-local residence duration may be subtracted as durations; the result is labeled transport/framework residual, not pure RTT.
- On CUDA workers, CUDA events in the same stream measure kernel duration, following the [CUDA asynchronous-execution timing guidance](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html). On Apple Silicon/MPS workers, `torch.mps.Event` provides the equivalent same-stream timing; on CPU-only workers, wall-clock spans around the kernel call are the only available measure and are labeled accordingly.
- Every process records chrony offset/error snapshots. If the measured bound is inadequate, precise one-way attribution is suppressed. PTP-capable hardware is a later option, documented by the [Linux PTP infrastructure](https://docs.kernel.org/next/driver-api/ptp.html).
- Impairment-profile contribution is established by paired shaped-versus-unshaped trials with all other factor hashes equal, not inferred from configured delay alone. RTT contribution has the stricter delay-only control in §13.3.

### 12.3 Integrity rules

`event_id` is globally unique; `event_seq` is strictly increasing per process start. Required relationships use stable IDs, never arrival order. Every real route has exactly one accepted terminal result or an explicit terminal failure. Attempts reconcile as accepted, rejected, expired, canceled, failed, duplicate, or late. Tensor row counts, byte counts, shapes, IDs, and segment offsets reconcile at every boundary. All locally nested spans are nonnegative.

Before any equality, admission, analysis, or claim check, the reader loads the schema bundle and proves the immediate registry has exactly 57 unique rows, the conditional table has two inactive-or-validly-activated rows, the byte/native ledger has 20 unique declarations, the runtime exclusion table has five rows, and the identity-name ledger has exactly 77 unique names. It rejects any type/registration/declaration overlap, missing classification, or double registration, then recomputes every structured artifact ID from the §3 projection and every byte/native identity from §11.1. Every literal `sha256` reference and every member of `input_hashes` or `factor_hashes` must resolve through its normalized URI to exactly one §17 row with the declared expected identity kind and registered type/declaration; a bare digest or mismatched locator is invalid. The run manifest's factor, placement-contract, evidence-child, and policy fields must equal those recomputed child IDs; nested child references are recursively checked and remain hash-sensitive. The manifest itself registers `none`, so its own recomputed ID is stored in the enclosing validation/evidence reference rather than inside `manifest.json`.

The verifier applies the §17 six-column index and two-root reachability rules before accepting completeness. A run is invalid if it has an identity projection/registry/declaration failure, unclassified identity name/path/evidence/schema, unresolved or sentinel ID, missing/duplicate/unreachable/dangling index entry, one-byte raw mutation, benchmark-critical `telemetry_drop`, missing router scores, unmatched spans, unresolved routes, factor-hash drift, or an absent topology proof. Operational exporters may reorder events, so event sequence and IDs remain mandatory; the OpenTelemetry file exporter, for example, gives no append-order guarantee ([OTel file exporter](https://opentelemetry.io/docs/specs/otel/protocol/file-exporter/)).

## 13. Exact metric definitions

All metrics publish unit, sample unit, inclusion filter, numerator, denominator, observation count `n`, and run/trial IDs. `MEASURE` excludes warmup but includes failures where stated. Let nearest-rank `Q_p(X)` sort `n` values ascending and return element at rank `ceil(p*n)` for `p in {0.50,0.95,0.99}`. Whether p99 is eligible for a primary claim, its minimum sample count, and any exploratory label are preregistered owner gates. `10,000` is only a proposed, nonbinding planning example and is not an acceptance threshold.

### 13.1 Core latency and throughput

For admitted request `r` with at least one received token:

```text
TTFT_r = ts(client_token_received first) - ts(request_send_start)

request_latency_r = ts(client_final_response) - ts(request_send_start)

ITL_(r,i) = ts(client_token_received r,i) - ts(client_token_received r,i-1), i>=2

tokens_per_sec_per_request_r =
  (output_tokens_r - 1) /
  seconds(ts(client_token_received last) - ts(client_token_received first)),
  only when output_tokens_r >= 2 and elapsed_seconds > 0
```

For a successful request with exactly one received output token, `tokens_per_sec_per_request_r=NA` with `exclusion_reason=ONE_TOKEN_NO_INTER_TOKEN_INTERVAL`; it is never evaluated as `0/0`. Every aggregate reports `eligible_request_count`, `one_token_na_count`, `zero_elapsed_na_count`, and `other_excluded_count` alongside `n`. If a stream chunk contains multiple tokens, record chunk latency and token count; do not invent per-token arrival timestamps. For each latency family report `Q_0.50`, `Q_0.95`, `Q_0.99`, `n`, and arithmetic mean, subject to the preregistered eligibility rules. System capacity over the common measurement interval `[t0,t1]` is:

```text
aggregate_output_tokens_per_sec = committed_output_tokens / seconds(t1-t0)
requests_per_sec = terminal_requests / seconds(t1-t0)
goodput_tokens_per_sec = output_tokens_from_successful_requests / seconds(t1-t0)
```

Canceled, rejected, timed-out, and failed requests remain in request outcome and offered-load denominators; successful-goodput is explicitly labeled.

### 13.2 MoE, queue, and load

For real, non-padding routed model tokens:

```text
logical_expert_accesses = count(route_decided.selected expert slots)

expert_accesses_per_model_token = logical_expert_accesses / routed_model_tokens

expert_accesses_per_token_layer = logical_expert_accesses / routed_token_layer_pairs

branch_elapsed_d = duration(dispatch_ready_d -> coordinator_response_received_d)

slowest_expert_branch_(token,layer) =
  max(branch_elapsed_d over accepted required logical expert batches)

slowest_expert_compute_(token,layer) =
  max(worker_compute_end_d - worker_compute_start_d over accepted required batches)

expert_queue_time_d = worker_queue_start_d - worker_queue_enter_d

load_imbalance_ratio = max(load_w) / mean(load_w over all eligible workers, including zero)

load_cv = population_stddev(load_w) / mean(load_w)

replica_busy_utilization =
  duration(union of accepted compute intervals for replica) / measurement_duration
```

`load_w` is reported separately for logical activation rows and useful device-compute nanoseconds. Padding, retry, duplicate, and canceled work are additionally reported as overhead rather than silently included as useful work.

### 13.3 Network and RPC

```text
packed_rpc_attempts_per_routed_token = all physical attempts / routed_model_tokens

logical_packed_rpcs_per_routed_token =
  distinct worker_layer_dispatch_id / routed_model_tokens

logical_expert_batches_per_routed_token =
  distinct logical_expert_batch_id / routed_model_tokens

wire_bytes_per_routed_token =
  sum(measured request+response wire bytes) / routed_model_tokens

logical_tensor_bytes_per_routed_token =
  sum(logical activation+output tensor bytes) / routed_model_tokens

cross_region_rpc_attempts_per_routed_token =
  attempts whose endpoints have different logical_region_id / routed_model_tokens

physical_wan_rpc_attempts_per_routed_token =
  attempts whose ordered topology link binds both endpoint physical_site evidence
  and a validated physical_link_evidence_id classified PHYSICAL_WAN
  / routed_model_tokens

directional_link_utilization =
  8 * measured_wire_bytes_on_link /
  (configured_link_bps * measurement_duration_seconds)

rpc_elapsed_d = duration(rpc_enqueue_d -> coordinator_response_received_d)

worker_residence_d = duration(rpc_received_d -> response_write_d)

transport_framework_residual_d = rpc_elapsed_d - worker_residence_d

impairment_profile_contribution =
  paired_metric(shaped_trial) - paired_metric(unshaped_trial)

rtt_contribution =
  paired_metric(delay_only_trial) - paired_metric(zero_added_delay_control)
```

`cross_region_rpc_attempts_per_routed_token` is a logical placement label and is never called WAN traffic. The physical-WAN aggregate is published only when `topology-proof.v1` validates the physical sites and ordered physical link for every required attempt; if any classification is unknown, the aggregate is `NA` with eligible/ineligible and evidence-missing counts. Emulated links, private-lab labels, and different logical regions do not satisfy that predicate.

The RTT pair changes only added directional delay. Physical nodes/links, bandwidth, jitter, loss, queue limits, workload, placement, scheduler, batching, software, warm state, and seeds have equal resolved hashes. The two topology objects differ only in preregistered directional delay fields—zero in the control and the named value in the treatment—and the comparison record proves field-wise equality everywhere else. Broader shaped-versus-unshaped deltas are named `impairment_profile_contribution`, because they may include bandwidth, jitter, loss, and queue effects. Directional utilization is also computed in fixed 100 ms windows and its maximum reported. Logical tensor, serialized payload, and measured wire bytes never share a field. The residual includes transport, gRPC buffering, staging outside worker residence, and measurement error; it is not called RTT. Configured delay alone is never reported as measured RTT contribution.

### 13.4 Locality and synchronization

Accepted-result locality classes are mutually exclusive:

```text
MACHINE_LOCAL: coordinator node_id == worker node_id
LAN_LOCAL: node differs, lan_domain_id equal
REGION_LOCAL: LAN differs, logical_region_id equal
CROSS_REGION: logical_region_id differs
```

For class `c`:

```text
accepted_locality_rate_c = accepted logical expert accesses in c / all accepted logical expert accesses
attempt_locality_rate_c = physical attempts in c / all physical attempts
```

Attempt locality keeps retries visible. A token-layer remote dependency round is the maximal set of parallel attempts released together whose required results must complete before that token-layer can proceed. Round `0` is the initial attempt set; after a failure or timeout, each sequential retry release is round `1..n` and adds another barrier:

```text
For successful eligible request r:
G_r = committed generated tokens
B_prefill_X,r = gating rounds during prefill satisfying locality predicate X
B_decode_X,r = gating rounds during decode steps producing tokens 2..G_r

all_phase_barriers_X_per_generated_token =
  sum_r (B_prefill_X,r + B_decode_X,r) / sum_r G_r

decode_only_barriers_X_per_decode_token =
  sum_(r:G_r>=2) B_decode_X,r / sum_(r:G_r>=2) (G_r - 1)

prefill_barriers_X_per_request = sum_r B_prefill_X,r / eligible_prefill_requests
```

`X` is published independently for different-node, logical-cross-region, and—only with complete physical proof—physical-WAN. The all-phase numerator explicitly includes prefill remote dependency rounds and amortizes them over generated tokens. The decode-only numerator excludes prefill and the prefill-produced first token. A one-token response is ineligible and its per-request decode-only value and any all-one-token aggregate are `NA`, never zero; publish eligible and ineligible request/token counts. Eligibility otherwise requires `MEASURE`, admitted successful requests, at least one generated token, a complete dependency graph and attempt outcomes, and no relevant telemetry drop.

A successful initial gating remote round contributes one. Parallel attempts released together count once even when several branches are remote; every sequential retry release contributes one additional barrier. All-local rounds contribute zero. `logical_event_id`, request/token/layer identity, `attempt_round`, release edges, and accepted/terminal outcomes make the count reconstructable without inferring from arrival order. Barrier count is not duration: also publish `prefill_barrier_wait_ns_per_request`, `all_phase_barrier_wait_ns_X`, and `decode_barrier_wait_ns_X`, with the same filters and separate queue/compute/RPC attribution.

Physical path classes form a separate proof-gated view, never an alias for the logical locality view:

```text
PHYSICAL_WAN: validated endpoint-site and ordered-link proof classifies WAN
PHYSICAL_NON_WAN: affirmative validated physical proof classifies non-WAN
UNKNOWN: proof missing, invalid, or inapplicable
```

For accepted expert accesses, the headline `physical_wan_hit_rate` and `physical_non_wan_hit_rate` are `WAN/(WAN+NON_WAN)` and `NON_WAN/(WAN+NON_WAN)` only when every required access is classifiable. If any required proof is absent, both headline rates and their aggregate are `NA`; publish `wan_count`, `non_wan_count`, `unknown_count`, `eligible_count`, and `ineligible_count`. An explicitly labeled eligible-only diagnostic may be shown but never substituted for the headline. Logical region difference, configured/emulated delay, and measured RTT never establish either physical class. A separate attempt view may expose retries.

### 13.5 Capacity, memory, and replication

```text
worker_busy_utilization =
  duration(union of all worker compute intervals) / measurement_duration

worker_useful_utilization =
  duration(union of accepted non-padding compute intervals) / measurement_duration

memory_utilization_time_average =
  integral_over_measurement(used_bytes / usable_bytes) / measurement_duration

replication_ratio = total resident exact expert-weight bytes / unique expert-weight bytes

replication_overhead_bytes = total resident expert-weight bytes - unique expert-weight bytes

locality_improvement =
  paired_accepted_local_hit_rate(replication) - paired_accepted_local_hit_rate(single_copy)

latency_improvement =
  paired_request_latency(single_copy) - paired_request_latency(replication)

storage_cost_per_locality_point =
  replication_overhead_bytes / (100 * locality_improvement), when improvement > 0
```

Accelerator utilization samples (CUDA `nvidia-smi`/NVML on Linux, `powermetrics`/Activity Monitor GPU counters on macOS) are diagnostic and reported separately from interval-union utilization because sampling cannot assign exact per-request work.

Placement-study outputs always prefix surrogate values with `predicted_` and runtime values with `observed_`. From `placement-feasibility-proof.v1`, plan/events, and held-out runs publish:

```text
constraint_violation_count = count(failed declared hard constraints)
worker_tier_overflow_bytes = sum_(w,t) max(0, resident_bytes_(w,t) - (C[w,t] - R[w,t]))
replica_budget_error_bytes = actual_resident_bytes - required_resident_bytes

predicted_worker_load_w = frozen service demand assigned to w
observed_worker_load_w = accepted rows and useful compute observed on w
predicted_load_imbalance = max(normalized predicted load) / mean(including zero workers)

assignment_churn = 1 - |A_new intersect A_old| / |A_new union A_old|
movement_bytes = sum bytes of assignments newly loaded on a worker
plan_instability = mean JaccardDistance(A_reference, A_perturbed)
```

Publish feasibility/violation counts and every lexicographic objective component; held-out TTFT, request latency, ITL, throughput/goodput, fan-out, RPCs, bytes, barriers, locality, queue/utilization, and predicted-versus-observed load; planning wall time/time-to-first-feasible, candidate/evaluation/iteration count, best objective by iteration, last improvement, deterministic stopping reason, seed variance, and perturbation instability; assignment additions/removals, changed worker sets, churn, moved bytes, startup load/verification/readiness duration and failures, and static replication overhead. Publish a relative optimality gap only when a verified bound exists. For immutable startup, in-service availability gaps, CAS conflicts, drain, rollback, forward repair, and requests affected during rollout are `NA`; they require executed Phase-4 fault evidence and are never inferred from the documented lifecycle.

### 13.6 Critical path

The analyzer constructs a request/token DAG from parent IDs and message pairs. Parallel expert spans are not summed. It reports mutually exclusive critical-path time for admission/cohort wait, shared prefill/decode compute, routing, dispatch batching, scheduler, coordinator serialization/staging, RPC residual, worker queue, cold load, expert device compute, response validation, combination, logits/sampling, and client streaming:

```text
unexplained_residual = measured_endpoint_latency - sum(critical_path_exclusive_categories)
reconciliation_error_rate = abs(unexplained_residual) / measured_endpoint_latency
```

The run declares a preregistered maximum reconciliation error; failing it blocks latency-breakdown claims. Expert-only timings are never presented as end-to-end latency.

## 14. Correctness and reference baselines

### 14.1 Baseline ladder

1. **Normative local reference:** unmodified `Qwen3MoeForCausalLM` from Transformers 4.51.3 with all experts local.
2. **Loopback distributed:** replacement adapter plus worker on the same host/device, with no shaping.
3. **Physical distributed:** selected coordinator/workers with the resolved topology.
4. **Instrumented ordered/FP32 accumulator:** optional diagnostic oracle, clearly not the unmodified reference.

The reference and candidate must share revision, tokenizer/chat/generation config, precision, prompts, batch shape/padding, attention backend, generation loop, seed, and output criteria. A single 80 GB CUDA GPU, or an Apple Silicon host with equivalent usable unified memory, is the clean reference preference, but availability is gated. Named multi-GPU or CPU-offload sharding is acceptable only as a distinct recorded baseline.

### 14.2 Machine-accounted baseline binding

Every correctness claim references one content-addressed `baseline-comparison.v1`, produced by `ResultAnalyzer` from immutable inputs:

```text
schema=baseline-comparison.v1, comparison_id, produced_at_utc
analyzer={version, binary_sha256,
          container_digest={uri, digest, expected_identity_kind=native_digest,
                            expected_type_or_declaration=RAW-07_OCI_IMAGES,
                            [retained_bytes={uri, digest, expected_identity_kind=raw_sha256,
                                             expected_type_or_declaration=RAW-07_OCI_IMAGES}]}}
reference={run_id, manifest_uri, manifest_sha256}
candidate={run_id, manifest_uri, manifest_sha256}
tolerance={artifact_uri, artifact_sha256, schema_version,
           preregistered_at_utc, preregistration_authority_id}
required_equal_fields[]
permitted_differences[]={field_path, reference_value_hash,
                         candidate_value_hash, preregistered_reason}
field_comparisons[]={field_path, reference_value_hash,
                     candidate_value_hash, outcome}
observations[]={observation_id, probe_id, request_id, token/layer/route coordinates,
                metric, reference_value_hash, candidate_value_hash,
                tolerance_rule_id, observed_delta, outcome=PASS|FAIL}
summary={required_equal_pass, unpermitted_difference_count,
         observation_pass_count, observation_fail_count,
         outcome=PASS|FAIL}
```

`comparison_id=sha256(JCS(record_without_comparison_id))` is the exact §3 projection alias for registry field `comparison_id`; all referenced artifacts remain in the projection and must re-hash before evaluation.

`required_equal_fields` contains at least model/checkpoint revision and weight digests, tokenizer/chat template/generation configuration, dtype/quantization, prompt-manifest hash, request order and batch/padding shape, attention backend, generation loop, seed schedule, output criteria, and every factor not explicitly named in `permitted_differences`. Permitted differences are a closed preregistered list—normally placement/directory/transport/topology fields needed to distinguish local reference from distributed candidate—and each requires a reason. An unexpected field difference fails the comparison.

The tolerance artifact must exist, be hashed, and be time-stamped before candidate outcomes are inspected. It names hardware class, probe/metric, coordinates or aggregation rule, exact versus floating comparison, `rtol`/`atol` or other bound, sample/repetition rule, and approval identity. A later tolerance artifact creates a new comparison; it cannot mutate or validate an earlier one. `correctness_validation=PASS` has no independent authority: the claim predicate passes only when this comparison record, all bound manifests, the tolerance artifact, and every required observation re-hash and reconcile.

### 14.3 Preregistered checks

| Probe | Required observation | Criterion |
|---|---|---|
| Manifest/loader | config/index/key inventory/digests | exact revision, all non-expert keys, all assigned expert tensors, shapes/dtypes/digests |
| Expert unit | pinned local MLP vs worker | exact identity/shape/order; floating tolerance; no NaN/Inf |
| Router unit | local block vs adapter on identical captured input | router logits close; selected IDs exact; normalized weights close |
| Binary round trip | before-send vs after-decode bytes | exact dtype/shape/row bytes/IDs/order |
| MoE loopback | unmodified block vs remote block | selected IDs exact; block output within preregistered envelope |
| Decoder/prefill/decode | layer states, KV, logits | same coordinates; tolerance by hardware class |
| Full generation | fixed deterministic prompts | entire greedy token-ID sequence exact |
| Cohort | same padded batch | active mask, router IDs, logits, tokens meet same criteria |
| Replica equivalence | two exact replicas | digest exact; output tolerance |
| Retry | exact-replica failure/retry | no duplicate contribution; same final correctness criteria |
| Missing expert | no eligible exact replica | explicit request failure, no approximate continuation |
| Failure suite | unavailable, mid-flight loss, slow, saturated, cold, late, reconnect | specified terminal outcomes; no corruption or double apply |

Exact checks cover model/digest, identities, shapes, dtype, cardinality, token coordinates, selected expert IDs, binary transport bytes, and greedy generated token IDs. Floating checks record `max_abs`, mean absolute error, relative L2, cosine similarity, and `torch.testing.assert_close`. PyTorch documents BF16 defaults of `rtol=1.6e-2`, `atol=1e-5`; these are a smoke-test starting point, not the full-model acceptance threshold ([testing docs](https://docs.pytorch.org/docs/stable/testing.html)). Hardware-specific thresholds are preregistered from repeated reference and loopback runs before placement/scheduler results are viewed.

PyTorch does not guarantee reproducibility across releases/platforms, and `index_add_` with colliding indices may be nondeterministic on CUDA and is not asserted deterministic on MPS either; the pinned block uses `index_add_` ([PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness)). Therefore bitwise heterogeneous BF16 equality is not promised. `torch.use_deterministic_algorithms(True)` may reject rather than fix the unmodified path. Repeated identical reference runs characterize its own envelope; `3–5` is only a proposed, nonbinding feasibility range, while the exact repetition count and acceptance rule remain owner gate 9. A changed selected-expert set for any real token-layer fails strict correctness, even if final tokens happen to match. Tolerances are never loosened after viewing a desired experimental result.

If routing is modified or forced, strict Qwen correctness is inapplicable; a separately preregistered quality/accuracy suite against unmodified routing and an activated `quality-evaluation-record.v1` become mandatory. Neither is selected or emitted by this design.

## 15. Workload and experimental methodology

The minimum workload suite includes:

- closed-loop interactive decode at concurrency 1;
- fixed-concurrency closed loop at named levels;
- open-loop Poisson arrivals at identical offered loads;
- deterministic burst/idle schedules;
- short/long prompt crossed with short/long generation, within the pinned context limit;
- a content-hashed representative text prompt manifest with provenance and license classification;
- separate prefill-heavy and decode-heavy summaries.

Synthetic expert-skew inputs belong only to replay/simulation-labelled studies.

Seeds derive independently from `(master_seed, domain_name, repetition_index)` for workload, sampling, scheduler, placement, network emulation, faults, and replay service. Changing one subsystem cannot shift another random stream.

Warm and cold runs are separate experiment cells. Warm runs start after all required experts are `DEVICE_WARM`, connections are established, configured warmup work completes, and a stability check passes. Warmup events remain stored with `phase=WARMUP` but are excluded from primary metrics. Cold runs restart processes/connections, reset caches and residency per an explicit procedure, and verify initial state before admission.

The project owner must preregister the independent repetition count, randomization/counterbalancing scheme, drift controls, estimator, uncertainty method and level, multiplicity policy, p99 eligibility/minimum, and acceptance thresholds before experimental outcomes are viewed. Five paired repetitions, a 95% bootstrap interval, and 10,000 p99 observations are proposed, nonbinding planning examples only. Whatever contract is approved, use identical prompts/arrivals/faults within pairs, report per-repetition paired observations, and never treat tokens in one run as independent experimental repetitions. Offered-load sweeps remain separate cells; policies are not compared at different self-selected loads.

The isolation validator freezes every factor except `vary_only`. Four mandatory control families are placement-policy-only, scheduler-only, topology-only, and execution-mode-only; batching-only is also explicit because batching changes kernel efficiency and queue evolution.

### 15.1 Sealed placement-policy comparison

`placement-comparison-contract.v1` is hashed before test unsealing. Every placement-only cell uses the same model/checkpoint/tokenizer/router/dtype/runtime and correctness contract; prompt IDs/order, generation limits, fixed open-loop arrivals (or separately labeled fixed-concurrency stratum), topology and discovery snapshot, physical allocation and service calibration, grouped partition, exact `constraint_set_sha256` (including candidate slots and exact replica-count/resident-byte budget), `objective_spec_sha256`, runtime `scheduler_spec_sha256` including candidate order/retry/deadline/tie rules, all three batching stages, fault plan, warm/cold state, reset/readiness procedure, master/domain seeds, telemetry/drop policy, environment/container, repetitions, estimator, thresholds, and multiplicity rule. Before comparison, every hash is recomputed through the same registered §3 projection; equality is checked on those recomputed IDs, not embedded placeholders, policy declarations, or serialized member order. Only `placement_policy_spec_id`, its derived complete feasible `placement_plan_id`, plan-derived directory rows/placement epoch, and resulting events/outcomes may differ.

The held-out test remains sealed until all candidate plans and the validation-selected comparator identities are hashed. Policy processes have no read access to test-cell telemetry. No objective weight, hyperparameter, replica budget, plan, or policy may update within a controlled cell or after test inspection; a change creates a new study and new held-out data. Missing or stale discovery starts a new snapshot and cell—it is never refreshed for only one policy.

Replay can guarantee the identical sealed route trace. `LIVE_E2E` cannot be supplied a forced trace while preserving the original-router claim: it uses identical inputs, arrivals, and seeds with the original router, then compares ordered generated token IDs and the complete ordered `(request, token_phase, absolute_token_position, layer, topk_slot, expert_id, normalized_weight)` route coordinates afterward. Any mismatch, correctness failure, telemetry loss, unplanned restart, or frozen-factor drift invalidates the placement-only pair. Endogenous queue/health outcomes may differ but never feed back into the plan.

Two-policy blocks use balanced `AB/BA`; six-policy blocks use a preregistered hashed balanced Latin-square or Williams schedule. Restore and verify the common start-state profile between runs. The independent unit is the paired block/run, never requests or tokens. Retain infeasible plans, timeouts, and failed runs as outcomes; do not silently replace one run. A replacement is a new complete block. Primary contrasts are optimized versus manual and optimized versus the strongest structured nonoptimized baseline selected only on validation; random feasible is summarized over its full preregistered seed set, not its best seed.

### 15.2 Phase-aware placement experiment matrix

`T0` means the validated homogeneous/LAN control; `T1` means a named validated heterogeneous and/or asymmetric impairment topology. `W-D`, `W-P`, and `W-M` are preregistered decode-heavy, prefill-heavy, and fixed-open-loop mixed workloads. `B0` is exactly 6,144 replicas and one replica per expert. `B_STATIC` is an owner-approved Phase-2B exact common replica-count **and** resident-byte budget variable; its value is intentionally unresolved.

| Campaign | Phase/mode | Required cells and policies | Permitted claim |
|---|---|---|---|
| `C0` | Phase 1 `LIVE_E2E` | `T0`, smoke workload, `B0`, manual explicit | correctness and tested-system behavior only |
| `DISCOVERY` | Phase 2A `LIVE_E2E` | neutral manual plan; train/validation prompts; pre/post application-path discovery | frozen discovery/routing/statistics inputs; no policy-effect claim |
| `L-ANCHOR` | Phase 2A `LIVE_E2E` | selected `T0/W-D` and `T1/W-M`, `B0`, all six policies, warm, paired/counterbalanced | held-out real policy effects in these exact cells after route equality |
| `L-GENERALIZE` | Phase 2A `LIVE_E2E` | remaining `T0,T1` × `W-D,W-P,W-M`, `B0`; optimized, manual, validation-selected strongest structured baseline | held-out real effects only for tested cells/policies |
| `L-COLD` | Phase 2A `LIVE_E2E` | selected `B0` cell; same confirmatory policies; cold reset | startup/load/verification/readiness outcomes, separate from warm steady state |
| `STATIC-REPL` | Phase 2B held-out `LIVE_E2E` | selected strata, exact common `B_STATIC`, frozen `FixedPrimary`, placement policy only | bounded static-replication placement effects; budget remains owner-gated |
| `R-VAL` / `R-TEST` | optional Phase 3 replay | all six on respectively permitted validation or once-unsealed test trace; frozen plans/calibration | screening within frozen replay inputs, never real performance |
| `L-RESILIENCE` | gated Phase 4 `LIVE_E2E` | owner-approved held-out fault and dynamic lifecycle cells | rollout/resilience only from executed validated cells |

Exact first-study cells, hardware, workload manifests, Phase-2B budget, repetitions, thresholds, and primary hypothesis remain owner gates. Replay is optional screening only after real execution and is not a Phase 1 or Phase 2 acceptance prerequisite. Only held-out validated `LIVE_E2E` cells can support real end-to-end performance or rollout claims.

## 16. Capture, replay, and simulation

Capture begins only from a correctness-qualified `LIVE_E2E` run. Full capture records router scores, selections, weights, causal route IDs, dependencies, actual expert inputs/outputs, and timing where configured. Capture overhead is measured by paired capture-on/off trials.

`CAPTURED_ACTIVATION_EXECUTION` sends recorded expert inputs through real RPCs and kernels and validates captured expected outputs. It proves the expert/transport seam, not causal generation. `FORCED_ROUTE_EXECUTION` with newly evolving activations is disabled by default because captured IDs need not be valid for changed hidden states; any use is explicitly instrumented and quality-tested.

`EVENT_REPLAY` has versioned timing variants:

- `OPEN_LOOP`: captured dispatch-ready times are fixed.
- `DEPENDENCY_DRIVEN`: token/layer events release only after the replayed predecessor completes.
- `CAPTURED_WORK_DAG`: actual captured-expert execution releases dependents inside the captured expert-only DAG.

When evaluating a scheduler or batcher, replay recomputes choices/batches from the recorded immutable decision inputs; it does not reuse captured decisions or batches. The event queue orders by `(logical_time_ns, event_type_priority, logical_event_id)` and stochastic draws use event-keyed seeds. Identical event-replay inputs must produce byte-identical decisions, batches, event order, and aggregates. Measured service profiles are frozen and keyed by worker, hardware/software revision, expert/class, dtype, row count, and residency tier; they cannot be transplanted silently across workers.

`ExecutionModeSpec` for replay additionally requires `replay_schema_version`, `replay_variant`; typed `{schema, uri, artifact_id}` child references for `run-manifest.v1`, `trace-manifest.v1`, `event-priority-table.v1`, `calibration-set.v1`, and `replay-completeness-profile.v1`; plus a `RAW-17` dependency-edge container reference whose rows carry registered `edge_id`s. A source run ID or mutable path without its indexed manifest identity is invalid.

Each captured dependency is one `replay-dependency-edge.v1` record:

```text
schema=replay-dependency-edge.v1, edge_id
source_run_manifest_sha256, source_trace_manifest_sha256
from_logical_event_id, to_logical_event_id
dependency_kind=DATA|CONTROL|RESOURCE_RELEASE|ATTEMPT_ROUND|TOKEN_CAUSAL
release_rule=ON_SUCCESS|ON_TERMINAL|ON_TIME|ALL_PREDECESSORS
predecessor_group_id, attempt_round, captured_offset_ns, attributes
```

Normatively, `edge_id = sha256(JCS(record_without_edge_id))`; this is the exact §3 projection alias for registry field `edge_id`, and every source/endpoint child reference remains hashed. Both endpoints must exist exactly once in the source trace; the graph must satisfy the selected replay variant's acyclicity/group rules; every non-root event declares all predecessors. A self-inclusive edge hash is invalid. The versioned `event-priority-table.v1` is fixed as follows for replay v1; changing it creates a new hashed table and replay identity:

| Priority | Event class | Tie behavior |
|---:|---|---|
| 10 | predecessor/resource release | ascending `logical_event_id` |
| 20 | scheduled fault/topology transition | ascending `logical_event_id` |
| 30 | service/RPC completion and terminal failure | ascending `logical_event_id` |
| 40 | request/token/layer admission-ready | ascending `logical_event_id` |
| 50 | scheduler and batching decision | ascending `logical_event_id` |
| 60 | dispatch/RPC start | ascending `logical_event_id` |
| 70 | logits/sampling/commit | ascending `logical_event_id` |
| 80 | client emission and aggregate/finalization | ascending `logical_event_id` |

Replay completeness is claim-specific and deny-by-default. All replay claims require manifest/trace/hash integrity, unique logical IDs, known event types, dependency closure, a bound priority table, complete decision contexts, seed domains, workload arrivals, placement/directory/topology epochs, attempt rounds/outcomes, and calibration keys. Scheduler comparisons additionally require every candidate/queue/health observation used by the policy; batcher comparisons require every compatibility field, row/byte limit, ready time, and sealing input; latency comparisons require all service-demand and link-profile keys plus causal edges. Missing inputs produce `replay_inputs_complete=FAIL`; they are never filled from current defaults.

For placement screening, replay uses one sealed trace for every policy, the same placement-study snapshot and service/topology calibration, exact common budgets, and the same frozen runtime scheduler. It must enforce the policy feature masks and no-online-update rule. Test replay occurs only after plans and comparator identities are committed; it does not become an optimizer input. Replay results remain in a separate table from `LIVE_E2E` results.

`DISCRETE_EVENT_SIMULATION` models computation/queues/links and supports comparative hypotheses only. Candidate rankings from replay/simulation must be checked against held-out real executions before they motivate an implementation claim. Only validated held-out `LIVE_E2E` can support real end-to-end performance or executed-rollout claims.

## 17. Immutable result store and visualization inputs

```text
results/
  schemas/{events,validation,baseline,topology-proof,replay-dependency,placement}.json
  experiments/<experiment_id>/
    experiment.json
    ARTIFACTS.sha256                       # canonical RAW-02 six-column identity inventory
    factors/{model,workload,hardware,topology,placement-study,placement-policy,placement,scheduler,batching,mode,faults,calibration}.json
    placement-study/{latency-discovery,routing-partition,activation-statistics,constraint-set,objective-spec,comparison-contract}.json
    placement-study/{discovery-cells,activation-statistics}.parquet
    trials/<trial_id>/attempts/<run_id>/
      manifest.json                         # complete run-manifest.v1
      directory/directory-snapshot.json     # directory-snapshot.v1 + replica-record hash
      placement/{plan,feasibility-proof,start-state,application-record}.json
      calibration/calibration-set.json      # calibration-set.v1 + bound data hashes
      status.json                           # attempt-status.v1
      events/event_type=<type>/node_id=<node>/*.parquet
      router_decisions/*.parquet
      resource_samples/*.parquet
      traces/{trace_manifest.json,logical_dispatches.parquet,activations/}
      replay/{dependency_edges.parquet,event_priority_table.json,completeness.json}
      proof/{topology-proof.json,baseline-comparison.json,validation-record.json}
      metrics/{request,token,expert,link}_metrics.parquet
      metrics/summary.json                  # metric-summary.v1
      visualization_inputs/
      logs/
      COMPLETE
```

Structured objects are resolved, registry-validated, projected, JCS-hashed, and written to immutable temporary names; raw/native identities follow §11.1. `ARTIFACTS.sha256` is byte-exact UTF-8 with no BOM, LF only, no blank lines, and exactly one header as its first line. In the grammar below, `<TAB>` is one byte `0x09`, `<LF>` is one byte `0x0a`, and braces delimit grammar metavariables; that notation is not stored. The fixed header is exactly:

```text
identity_kind<TAB>type_or_declaration<TAB>normalized_uri<TAB>digest<TAB>byte_count<TAB>parent_uri<LF>
```

Each following data row is exactly:

```text
{identity_kind}<TAB>{type_or_declaration}<TAB>{normalized_uri}<TAB>{digest}<TAB>{byte_count}<TAB>{parent_uri}<LF>
```

The header literal appears once; data rows contain exactly six nonempty fields, contain no literal TAB, CR, or LF within a field, and the file ends with the last row's LF. `identity_kind` is the case-sensitive lowercase token `structured_jcs`, `raw_sha256`, or `native_digest`. A structured `type_or_declaration` is the exact canonical registry token, which is lowercase where defined; a byte/native value is the exact case-sensitive `RAW-01`–`RAW-20` declaration token and is never case-folded into an alias. Data rows are in strictly increasing lexicographic order of `normalized_uri` UTF-8 bytes. Any extra header, header mutation, comment, blank row, extra/missing column, alternate delimiter, CRLF, BOM, invalid UTF-8, out-of-order row, or trailing byte is invalid.

Both URI fields use one canonical relative URI-path encoding. Starting from a strictly decoded UTF-8 logical path, normalize Unicode to NFC, use `/` as the only separator, and percent-encode each segment's UTF-8 bytes outside RFC 3986 unreserved `[A-Za-z0-9._~-]` as uppercase `%HH`; percent-encoded input is decoded exactly once and re-encoded by that rule. Reject invalid escapes or UTF-8, an absolute path or URI scheme, leading/trailing slash, empty segment, decoded `.` or `..` segment, backslash, query/fragment syntax, NUL/control character including TAB/CR/LF, an encoded separator/control, symlink or hard-link alias, and two spellings that normalize to the same URI or physical target. Sorting and equality occur only after this normalization.

For `structured_jcs` and `raw_sha256`, `digest` is exactly `sha256:` followed by 64 lowercase hexadecimal characters. Their `byte_count` is the exact stored-file length as unsigned base-10 ASCII: `0` for zero and otherwise no leading zero. A `native_digest` row instead has exactly `digest=<registered-native-algorithm>:<canonical-lowercase-native-digest>` and `byte_count=-`; the lowercase algorithm token and native-digest grammar are fixed by its `RAW-01`–`RAW-20` declaration, and the first colon separates them. A native row never contains a retained-byte hash, delimiter-separated list, JSON, blank component, or composite digest. If claim-relevant bytes are retained, they have a different normalized retained-copy URI and one distinct `raw_sha256` row with `digest=sha256:<64-lowercase-hex>`, a decimal byte count, and the same owning structured `parent_uri`; the structured parent explicitly references both rows. One physical file or URI cannot be indexed as both native and raw evidence, and neither row is double-counted.

Exactly two data rows use the literal `parent_uri=-`: the `structured_jcs` rows for `run-manifest.v1` and `validation-record.v1`. Every other structured, raw, or native row has exactly one normalized `parent_uri` that resolves to an indexed `structured_jcs` owner. Additional typed references do not create another index parent or row. The verifier rejects a third root, either missing root, `-` on any other type, a nonexistent or non-structured parent, duplicate child URI, multiple parent rows, self-parent, cycle, or node whose unique parent chain does not terminate at one of the two roots. Inline subobjects and individual Parquet/event rows have no line. Every material file has exactly one line except `ARTIFACTS.sha256` itself, whose hash and byte count occur once in the detached `RAW-02` publication-root tuple because it cannot self-list; `COMPLETE` has one `RAW-03` line owned by `attempt-status.v1`.

The verifier recomputes every line, rejects a mismatched identity kind/type/declaration, unregistered semantic JSON, double registration, undeclared container, byte-count drift, or referenced child absent from the index, and checks every `sha256`, `input_hashes`, `factor_hashes`, and each of the four native container locators against the unique expected row. For each native and optional retained locator, normalized URI, copied `digest`, identity kind, declaration, and structured parent must all equal that row; a matching URI with a mismatched digest is invalid. The copied child digests remain unchanged in ordinary parent JCS and recursive ancestor recomputation. It then recursively walks typed child edges from both recomputed roots. The run-manifest walk must reach every execution/configuration input and authoritative run output; the validation-record walk must reach the run manifest, schema/claim/completeness rules, raw-event manifest, topology/baseline/tolerance evidence, analyzer/environment identity, and every artifact used by an allowed claim. Their union must reach every indexed authoritative/claim/evidence row; diagnostic-only `RAW-20` rows may remain outside claim reachability but must still have one `attempt-status.v1` parent. No referenced input may be dangling or unindexed, and no indexed authoritative artifact may be unreachable. The closure scanner also requires exactly the 57/2/20/5/77 ledger counts and fails any unclassified new field, evidence kind, path class, or schema before `COMPLETE`.

`COMPLETE` is atomically published only after the detached index identity, manifest, validation graph, and all referenced rows reconcile. Attempts are never overwritten, including failures. Parquet supplies typed columnar storage and inspectable schema metadata ([Apache Arrow Parquet docs](https://arrow.apache.org/docs/13.0/python/parquet.html)). The MVP uses local artifacts and a deterministic verifier, not a distributed result or hashing service.

Required visualization inputs are:

1. latency-breakdown critical-path exclusive spans by request/token/category;
2. expert path edges `(request,token,layer,expert,replica,node,physical_site,logical_region,attempt,latency)`;
3. fixed-interval worker busy/queue/GPU/memory series;
4. expert popularity by layer/expert/split/prefill-decode stratum with numerator, denominator, and request-cluster interval;
5. same-layer and cross-layer descriptive co-activation/lift inputs with split provenance; cross-layer charts disclose zero direct placement-distance weight;
6. machine/LAN/region/cross-region accepted and attempt hit rates;
7. a **separate** physical-WAN/non-WAN accepted-access view with `PHYSICAL_WAN`, `PHYSICAL_NON_WAN`, and `UNKNOWN`; when any required proof is absent its headline/aggregate is `NA` and it shows WAN/non-WAN/unknown plus eligible/ineligible counts, never logical cross-region substitution;
8. directional discovery matrix with p50/p95/p99, interval, age, coverage, drift, failure rate, payload/concurrency class, and measured-underlay versus measured-under-emulation authority;
9. placement feasibility slack and objective components, predicted-versus-observed load/fan-out/cost, solver convergence/valid gap, seed variance/instability, assignment churn/movement, and startup application/readiness/failure inputs;
10. diagnostic queue CDF, batch-size distribution, retry/failure timeline, topology probe comparison, and placement-memory map.

Plots are derived artifacts; raw events and declared filters remain authoritative.

## 18. Deployment choices

| Option | Evidence ceiling | Use and decision |
|---|---|---|
| One host, processes/namespaces | software/RPC/shaping preflight; no physical-node/geography claim | Phase 0 only |
| 2–4 physical Mac and/or Linux hosts (NVIDIA CUDA, Apple Silicon/MPS, or CPU) with scripts/systemd/launchd/Compose | physical-node, heterogeneous-backend, and controlled emulation evidence | **proposed, nonbinding Phase 1 shape; owner approval required** |
| Same-region GPU VMs | physical-node evidence with cloud variance | proposed alternative; owner approval required |
| Real cloud multi-region | named physical-link results | Phase 4 only if geography is required |
| Kubernetes | same scientific ceiling plus CNI/orchestration variables | defer until repeated operations failures justify it |
| Large-scale consumer-hardware fleet (beyond the small local cluster) | heterogeneous backend evidence at scale | Phase 4; the small local Mac/Linux cluster itself is in scope from Phase 1 |

No Phase 1 transport security policy is selected here. Plain TCP on an isolated trusted network and authenticated mTLS/VPN are candidate profiles, not defaults. Before deployment, the project owner must approve the physical environment, exposure boundary, threat model, identity/authentication/authorization, encryption, certificate or VPN lifecycle, and whether security overhead belongs in the measured path. The selected policy becomes a content-addressed factor and entry gate; a `PRIVATE_LAB_V1` label alone cannot establish safety. Independently, the proposed container least-privilege posture gives narrowly scoped GPU access and `NET_ADMIN` only to the topology agent on Linux/NVIDIA nodes, subject to the same security approval; full privilege is not a default ([NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), [Docker capabilities](https://docs.docker.com/engine/containers/run/)). macOS hosts run the worker as a native, non-containerized process instead, because Docker Desktop on macOS cannot pass Metal/MPS through to a container; the equivalent least-privilege posture for a native macOS process (user/group scoping, sandbox profile, no elevated network capability) remains the same class of owner-approved security gate.

UCX/TCP, NCCL, RDMA/GPU-direct, libp2p, and custom sockets are deferred until profiling shows transport overhead is the limiting gap and the replacement has a validated shaping/accounting path, with libp2p specifically taken up as Phase 2 transport research rather than left an open-ended "later." No transport substitution inherits the gRPC/tc evidence label automatically.

## 19. Phased roadmap and module plan

### Phase 0 — feasibility and reference spike

**Entry:** checkpoint access and candidate hardware (NVIDIA CUDA, Apple Silicon/MPS, or CPU).  
**Modules:** `model_manifest`, `reference_runner`, `partial_loader_spike`, `expert_kernel`, `tensor_codec`, `grpc_loopback`, `topology_probe`.  
**Exit:** exhaustive shared/expert inventory; expert equivalence; measured memory fit/headroom; exact binary round trip; validated directional shaping and application-path discovery feasibility; preregistered prompts/tolerances.  
**Pivot:** if the model does not fit, change hardware or approve a separately named smaller MoE. Never silently quantize, mock experts, or use replay while retaining the Qwen correctness claim.

### Phase 1 — real distributed correctness MVP

**Entry:** every Phase 0 gate and the owner-approved deployment/security policy pass.  
**Modules:** `api_adapter`, `cohort_executor`, `remote_moe_block`, `expert_directory`, `explicit_placement`, `fixed_primary`, `dispatch_batcher`, `expert_transport`, `expert_worker`, `topology_controller`, `event_writer`, `result_store`, `correctness_analyzer`. Phase 1 implements manual explicit placement through the immutable startup states in §6.2, one atomic `DirectorySnapshot`, and worker-incarnation rejection only; no optimizer, replay prerequisite, or dynamic prepare/load/commit/drain exists. A minimal two-placement comparison reuses the `baseline-comparison.v1`/latency-reconciliation machinery already required for correctness; it does not require the Phase 2A offline planner, statistics pipeline, or sealed snapshot.  
**Exit:** physical-host `LIVE_E2E`; every route terminates; reference tensor/logit/token checks pass through `baseline-comparison.v1`; actual-path topology proof passes; `validation-record.v1` allows only its proven claims; latency reconciliation and immutable reproduction pass; and at least two manual explicit placement configurations have been exercised on identical hardware/topology/workload with their directional latency and correctness deltas reported, establishing placement as a measurable factor.  
**Pivot:** any semantic mismatch blocks performance work; any coordinator bottleneck is first measured and reported, not hidden by redesign.

### Phase 2A — placement-optimization and transport research campaign

**Entry:** Phase 1 remains green, including its two-placement latency/correctness comparison.  
**Modules:** active application-path discovery, grouped routing partitions/statistics, sealed placement-study snapshot, offline policy implementations/feasibility proof, immutable startup application, workload driver, comparison/isolation validator, statistics, visualizations, a libp2p (or other peer-to-peer protocol) transport prototype benchmarked against the Phase 1 gRPC baseline under the same shaped topology, and critical-path bottleneck analysis over the accumulated telemetry. These are coordinator scripts and immutable artifacts, not services.  
**Exit:** exactly one replica per expert; all six placement policies have plans committed before held-out unsealing; `L-ANCHOR`, `L-GENERALIZE`, and `L-COLD` validated `LIVE_E2E` cells pass correctness, route equality, fairness, uncertainty, and full metric derivation, converging on a recommended placement strategy; a documented, evidence-based recommendation on libp2p (or another P2P protocol) versus gRPC for the eventual decentralized dAI architecture; and a ranked list of measured critical-path bottlenecks in the distributed data plane.  
**Pivot:** add only the mechanism needed to resolve an observed experimental limitation.

### Phase 2B — bounded static replication

**Entry:** Phase 2A shows a named replication hypothesis and the owner approves the exact common replica-count/resident-byte budget and cells.  
**Modules:** the existing offline planner and immutable startup loader only; runtime `FixedPrimary` remains frozen.  
**Exit:** fixed-budget static-replication placement comparisons pass the same held-out `LIVE_E2E` contract. Alternate schedulers remain separate experiments. No concrete budget is selected by this document.

### Phase 3 — calibrated replay

**Entry:** at least one validated real trace plus measured topology/worker calibration.  
**Modules:** activation capture/execution, deterministic replay engine, calibration validation.  
**Exit:** proof labels pass; repeat replay is byte-identical; candidate rankings are checked against held-out real executions; disclosed error is acceptable for the intended screening use.  
**Pivot:** if replay fails to predict held-out results, narrow its claim/use rather than adding unsupported fidelity.

### Phase 4 — external validity and resilience

Add real multi-region deployments, broader churn, warm/cold movement, consumer hardware, or optimized transports only for named hypotheses. If dynamic placement itself is owner-approved as a hypothesis, this is the earliest phase that may implement and execute the documentation-only `PREPARE -> LOAD_ADDITIONS -> VERIFY_COMPLETE -> READY -> CAS_COMMIT -> DRAIN_REMOVALS -> UNLOAD -> FINALIZE` lifecycle. Continuous adaptation, periodic refresh, rollback automation, and rollout claims remain gated until executed fault evidence exists. Each extension has its own baseline and evidence label.

### Phase 5 — operational platform if justified

Kubernetes, automated provisioning, durable shared services, dashboards, and a richer control plane require recorded recurring configuration, recovery, allocation, or campaign-volume failures that scripts and immutable artifacts cannot solve.

## 20. Simplicity review and alternatives

The design uses the narrowest semantic seam: the pinned Transformer runtime remains the model oracle, and only sparse expert execution crosses RPC. It uses one coordinator, in-memory directory snapshots, static config, deterministic controls, one transport, Linux shaping, append-only local artifacts, generated plots, and an offline placement script over sealed files. Discovery, planning, feasibility checking, and startup application add no long-running control-plane service.

Deferred alternatives have explicit triggers:

| Alternative | Why not MVP | Revisit trigger |
|---|---|---|
| stock vLLM/SGLang coordinator | arbitrary heterogeneous per-replica WAN dispatch is not a stock contract and obscures the seam | correctness is stable and shared-path throughput is measured limiting |
| remote whole MoE layer | hides per-expert placement/replication | never for experiments requiring per-expert control |
| one RPC/token or expert | creates avoidable overhead | diagnostic protocol comparison only |
| continuous batching/cache compaction | adds cache/lifecycle complexity | fixed cohorts are a measured serving bottleneck |
| dynamic migration | unnecessary for static placement questions | migration is itself the hypothesis |
| UCX/NCCL/RDMA/custom sockets | complicates churn and may bypass qdisc | gRPC transport dominates and new shaping is validated |
| libp2p / other P2P transport | would conflate transport-protocol validation with the Phase 1 correctness proof on a fixed, fully-known topology | Phase 2 transport research, benchmarked against the gRPC baseline for the eventual decentralized dAI architecture |
| Kubernetes/distributed directory/store/dashboard | adds operational variables | repeated campaigns prove scripts/local artifacts insufficient |
| replay/simulation first | cannot prove Qwen correctness | never before Phase 1 |
| online placement learning/continuous discovery | leaks outcomes and changes the treatment | only after a separately approved adaptive-policy study |
| custom solver service | offline deterministic planning is sufficient | only if measured campaign volume cannot be handled by scripts/artifacts |

## 21. Risk register, security, and retention

| Risk | Likelihood/impact before evidence | Detection | Mitigation/gate |
|---|---|---|---|
| model/partition does not fit | high/high | exhaustive load and headroom measurement | change hardware or explicitly rename baseline/model |
| remote seam changes semantics | medium/high | per-boundary differential probes | block Phase 1 on any unexplained mismatch |
| numerical drift changes routes/tokens | medium/high | repeated route/state/logit/token comparison | hardware envelopes; exact route/token gates |
| coordinator bottleneck | high/high | critical path and saturation | measure; decentralize only if it masks study variable |
| batching confounds studies | high/high | factor hashes and batch telemetry | freeze `batching_id`; isolate batching studies |
| shaping differs from config | medium/high | read-back, counters, directional payload probes | invalidate topology claims |
| wrong traffic shaped | medium/medium | address/filter/counter audit | separate management/data paths |
| clocks corrupt attribution | high/medium | uncertainty/negative span checks | local monotonic spans; suppress one-way claims |
| retries double-apply | medium/high | fault probes and accepted-result reconciliation | stable logical IDs; one accepted result |
| trace overhead/volume | medium/medium | paired overhead and storage-rate trials | full correctness capture first; later sampling only when declared |
| replay mistaken for execution | high/high | mechanical claim allowlist | block report publication |
| workload bias | medium/high | preregistered stratified suite | report strata before aggregates |
| placement test leakage/circular tuning | high/high | feature-access audit, test seal/unseal chronology | hash all plans/comparators first; new held-out data after any post-test change |
| stale or asymmetric discovery | high/high | coverage/age/interval/drift and directional-cell checks | fail admission; create a new common snapshot, never per-policy imputation |
| infeasible/unstable plan | medium/high | complete constraint proof, repeated deterministic solve, perturbation analysis | never load partial plan; report instability and narrow claims |
| placement/scheduler coupling | high/high | forbidden-plan-field and frozen-factor audits | one copy + `FixedPrimary` in Phase 2A; separate later factors |
| physical-WAN misclassification | high/high | site/link proof completeness and unknown-class counts | headline `NA`; never substitute logical region or emulation |
| startup result overstated as rollout | medium/high | phase/event/claim checks | dynamic metrics `NA` until executed Phase-4 fault study |
| weights/prompts exposed | likelihood gated by owner-selected deployment; high impact | exposure scan/access audit | approved trust/security profile before deployment or capture |
| cloud cost/scarcity | medium/medium | budget/quota record | emulate first; selected physical validation only |

Security gates before any deployment are an owner-approved threat/trust model and explicit decisions on authenticated coordinator/worker identities, authorization for load/execute calls, encryption in transit, secret management, input/output log redaction, tenant isolation, denial-of-service limits, dependency/image scanning, and incident/credential-rotation procedures. Applicability may differ for a truly isolated lab, but the document does not waive or select these controls. Required controls and their overhead must be measured on the actual path.

Retention is unresolved. Before capturing prompts, full router scores, activations, or outputs, the owner must approve data classification, consent/provenance, encryption at rest, access list, retention duration, deletion procedure, and whether tensors can reconstruct sensitive text. Immutable scientific attempts do not override privacy deletion obligations; a deletion tombstone and audit record must distinguish policy removal from experimental corruption. No deletion-audit evidence may be emitted until the conditional `deletion-audit-record.v1` contract is activated in a new closed registry; this registration gate chooses no retention or deletion policy.

## 22. Non-goals and unresolved owner decisions

### MVP non-goals

- Dynamic migration, continuous placement adaptation, or router-learning research.
- Production-grade global reliability, autoscaling, service mesh, or orchestration.
- Consumer/Mac backends, multiple runtimes, or every scheduler/placement algorithm.
- Custom/RDMA transport or peak serving optimization.
- Perfect WAN reproduction or geography claims from logical labels/emulation.
- Distributed configuration/directory/result databases or polished dashboards.
- Quality benchmarking while routing remains identical, except the required correctness comparison.
- Forced routing for placement comparisons, policy-specific discovery, held-out/test features in planning, or online updates within a cell.
- A service-heavy placement controller, automatic snapshot refresh, migration daemon, or replay/simulation prerequisite.

### Unresolved gates

1. Reference accelerator (CUDA GPU or Apple Silicon/MPS) availability and aggregate worker memory/headroom.
2. Exact PyTorch, accelerator backend/version, driver (where applicable), container, attention backend, tokenizer, and generation revisions.
3. Hardware-specific tensor/logit tolerance envelopes.
4. Exact OpenAI-compatible fields, SSE semantics, cancellation, deadlines, and concurrency subset.
5. Cohort/admission, dispatch/RPC, queue, microbatch, and maximum-context limits; inactive-row versus cache-compaction choice.
6. Initial physical deployment and whether it is trusted/private or includes mTLS/VPN overhead.
7. Achievable clock uncertainty and whether any one-way timing claim is permitted.
8. Sustainable router-score/activation capture rate and acceptable telemetry overhead.
9. First primary hypothesis, exact placement-study cells and workloads, Phase-2B common replica/byte budget, preregistered repetitions, calibration/discovery tolerances, statistical thresholds, and p99 minimum.
10. Whether initial emulated evidence suffices or physical multi-region validation is required; no physical-WAN or rollout claim is authorized by this design alone.
11. Dataset/prompts provenance, licensing, privacy classification, and retention/deletion policy.
12. Hardware-fit fallback: acquire capacity or explicitly approve a separately named smaller/quantized baseline.

## 23. Validation plan

Every check is phase-tagged; a later replay or rollout check is never a Phase 1 prerequisite.

1. **P0/P1:** inventory every checkpoint tensor and prove coordinator plus worker coverage/digests.
2. **P0/P1:** run expert, router, block, layer, prefill, decode, cohort, and full-generation differential probes.
3. **P0/P1:** prove packed binary RPC shape/dtype/order/identity and coordinator-only weights.
4. **P1:** exercise unavailable replica, mid-flight loss, slow worker, saturation, cold load, retry, duplicate/late, reconnect, and no-replica behavior.
5. **P0/P1:** prove actual-path shaping by rule read-back, byte counters, and independent directional delay/bandwidth/loss probes.
6. **P1:** validate queue, attempt, logical locality, dual barrier fixtures, capacity, and critical-path reconciliation from raw events. Fixtures cover prefill, one-token decode-only `NA`, normal decode, a successful initial round, parallel branches, and sequential retries, plus prefill/request and wait duration outputs.
7. **P0/P1/P2/P3:** validate exact identity closure: 57 unique immediate registry rows (31 prior plus the specified 26), two conditional contracts absent unless validly activated before gated emission, 20 unique raw/native declarations, five runtime-only exclusions, and 77 unique classified hash/digest/artifact-key names, including literal `sha256`, `input_hashes`, and `factor_hashes`. Require exactly one `own_identity_field` or `none` per structured type and no structured/raw/runtime double registration. For every embedded or out-of-band artifact ID in the resolved bundle, six-column `ARTIFACTS.sha256`, run manifest, directory, calibration, placement, proof, baseline, replay, and validation graph, recompute `sha256(JCS(resolved_object_without_its_registered_own_identity_field))`—or the complete object for `none`—and require exact equality. Resolve every polymorphic `sha256` and each aggregate input/factor member by normalized URI to exactly one index row and require its expected kind, type/declaration, and value to match; reject a bare digest, absent URI/type, kind inference, duplicate/missing row, and correct digest bound to the wrong artifact. Negative fixtures reject a missing registry/type/declaration; duplicate row/URI/physical target; wrong, aliased, nested, or multiple own-field registration; absent registered field; self-inclusive hash; blank, `null`, all-zero, `TBD`, `GATED_*`, or other sentinel identity; blanking instead of deleting; semantic JSON mislabeled raw/opaque; unclassified new identity name/path/evidence/schema; and conditional quality/deletion evidence without activated contracts. Verify `record_without_validation_record_id`, `record_without_topology_proof_id`, `record_without_comparison_id`, and `record_without_edge_id` are exact aliases of the global projection. Byte fixtures exercise UTF-8/LF/no-BOM and exactly one literal header; true TAB delimiters; exactly six nonempty fields; base-10 counts; canonical kind/type case; lowercase digests; URI percent-decode/re-encode, NFC, alias rejection, normalized-URI byte sorting, and final LF. Reject BOM/CRLF/invalid UTF-8; raw TAB/CR/LF or controls in fields; malformed/lowercase/noncanonical escapes; encoded separator/control; absolute/scheme/dot/empty/backslash/query/fragment paths; duplicate normalized URI; disorder; extra header/comment/blank/trailing byte; and extra/missing column. Require exactly the `run-manifest.v1` and `validation-record.v1` structured rows to have `parent_uri=-`; inject either missing root, a third root, wrong-type root, missing/non-structured/self/multiple parent, cycle, and non-rooted component and require rejection. For every `RAW-01`–`RAW-20` media/encoding class, flip, add, or remove one byte and require digest/byte-count/parent failure. Native fixtures require a registered lowercase algorithm and canonical lowercase native digest with `byte_count=-`; reject unknown/case-varied algorithms, invalid native grammar, a byte count, blank/composite/JSON digest, or appended retained-byte hash. When bytes are retained, require a different normalized URI, distinct `raw_sha256` row with `sha256:<64-lowercase-hex>` and decimal count, the same structured parent, and explicit parent references to both; reject same-URI/physical-file native/raw double counting, missing half, mismatched parent, and mutation of either the native reference or retained bytes. Instantiate all four native container locators—the validation analyzer, model runtime, run container fingerprint, and baseline analyzer—with the exact four-member primary shape and optional four-member retained shape. Resolve normalized URI, copied digest, kind, `RAW-07_OCI_IMAGES` declaration, and parent to distinct exact rows; reject any URI/digest mismatch even when URI resolution succeeds. For each of the four field paths, mutate only the native digest at stable URI and then only the retained digest, requiring the direct parent and every ancestor to change; omission of retained evidence remains valid and makes no retention claim. Parse the three complete synthetic `model-spec.v1` fixture objects from §11.1, require registered `own_identity_field=none` and absence of `model_spec_id`, JCS-canonicalize the complete object, and independently reproduce `4ed18e126cd11687b25abbae48a7492536a4cd42b65e7cff6551397db976df3b`, `ea12f2fda667d563431de2147c01957dc639260901923d31cc9e1b1eaa251d33`, and `784704fb8c6a5e2f18aec453ccb46d7dd5f3e75a3d9536b7d8c7881408a171f0`. Require the native and retained objects to differ from base at exactly their one named digest value. Parse the exact noncanonical base input, JCS-canonicalize it, and require the base ID; direct source-byte hashing is invalid. Reject fixture omissions, extra fields, alternate values, `model_spec_id`, ellipsis/repetition shorthand, placeholder/default substitution, pre-JCS hashing, BOM, whitespace, or trailing LF in the hashed bytes. Reject missing, blank, sentinel, malformed, uppercase, or wrong-length digest; scalar/composite encoding; wrong kind/type/declaration; dangling, aliased, or duplicate URI; same row/file for native and raw; unequal parents; row-ID or dereferenced hashing; and alternate fields such as `row_id`, `native_digest`, or `retained_digest`. Re-run the two-root recursive walk and prove the counts remain exactly `57/2/20/5/77`; the fixed `digest` member is an index-value copy, not a new outer semantic identity name. Audit every literal occurrence of the former count 74 and require it to be explicitly historical, revision-7-only, and superseded by current 77 rather than an active fixture or acceptance clause. Child-sensitivity changes one referenced child ID and must change every ancestor; serialization-order permutations must preserve every JCS ID. Prove inline/container distinctions, replay row/container dual identities, one schema-bundle ownership, authoring-YAML absence, and constraint-leaf registration-or-complete-inline exclusivity. Recursively walk from recomputed `run-manifest.v1` and `validation-record.v1`, requiring every indexed authoritative/claim/evidence artifact reachable, every child indexed once, no dangling reference, and no unindexed claim input. Then recreate topology, manual placement, workload, environment, seeds, and mode labels and reconcile `RunManifest`, `DirectorySnapshot`, `CalibrationSet`, topology proof, baseline comparison, analyzer/evidence, and `validation-record.v1`. Inject raw `PASS`/`VALID`, unknown claims, missing evidence, and hash drift and require denial.
8. **P2A:** validate every joint application-path `L_rpc` cell's complete source/coordinator, candidate-slot endpoint, transport/TLS, paired request/response payload, concurrency, network-condition, discovery-snapshot, and uncertainty/coverage key; verify coordinator-monotonic round-trip provenance, coverage, age, sample count, confidence width, drift, loss, topology authority, joint interpolation domain, and pre/post stability. Missing, stale, under-covered, mismatched, or out-of-domain cells must return the corresponding structured infeasibility with no one-way decomposition, extrapolation, configured-delay imputation, or partial plan.
9. **P2A:** recompute all grouped train/validation frequency, routing-mass, same-layer co-activation/lift, and cross-layer co-activation rows including numerators, denominators, eligible/excluded counts, strata, intervals, and leakage audit; prove the exact `e<e'` and ordered positive-offset domains. Fixtures include an empty stratum, an absent expert/zero marginal, zero total routing mass, no eligible paired-layer observation, diagonal/symmetric input attempts, and `NA` ingestion; require typed `NA` with counts and reject every `NA`-as-zero path. Prove held-out test routes/outcomes were sealed and absent from every policy input.
10. **P2A/P2B:** independently recompute every `C_hat_g` and lexicographic objective tuple from the content-addressed discovery/calibration/objective artifacts and recorded exact lookup keys, confirming one and only one `L_rpc` use per required slot branch, the exact keyed `S` value, and explicit `P_coord=0`; reject missing/stale/under-covered/out-of-domain cells and any intercept allocation. Exercise feasible/infeasible per-slot occupancy and compatibility, worker/tier capacity/reserve/slot limits, replica bounds, exact budgets, failure-domain caps, resilience sets, and later transition-headroom fixtures; reconcile any certificate/bound.
11. **P2A/P2B:** seal one `constraint-set.v1`, require its root and every leaf-table hash equal across all policies, and reject attempts to add/remove/relabel/reorder slots, alter compatibility/capacity/reserve/replica/failure/resilience/budget inputs, or reference a different universe. Solve identical inputs repeatedly and require byte-identical assignment vectors. A negative fixture permutes placement-plan rows, directory insertion order, and JSON/Parquet serialization order and requires both surrogate and runtime `FixedPrimary` to select the same canonical replica; a control change to the hashed candidate-order rule must change `scheduler_spec_sha256` and invalidate a placement-only pair. Reject forbidden plan fields for primary/fallback order, health/queue, attempts, dispatch, or scheduling. Phase 2A additionally requires exactly 6,144 one-copy assignments; Phase 2B uses only the owner-approved exact common budget.
12. **P1/P2:** exercise every immutable startup state and inject hash, digest, capacity, residency, and incarnation failures before commit; require no directory publication and admission closed. Confirm discovery freshness is an admission predicate, not a live directory refresh.
13. **P2A/P2B:** validate the six policy feature masks and roster; equality of common snapshot/runtime/workload/topology/discovery/calibration/constraint-set/objective-spec/scheduler-order/budget/batching/fault/start-state/seed/telemetry/repetition hashes; permitted derived differences only; test-seal chronology; no online updates; paired order; and post-run complete token/route equality.
14. **P1/P2:** run paired delay-only and broader impairment-profile controls. Independently verify the physical-WAN/non-WAN view reports headline/aggregate `NA` plus eligible/ineligible counts on any missing physical proof while logical cross-region remains derivable.
15. **Optional P3 only:** capture a correctness-qualified real trace, validate IDs/order/completeness, require `edge_id = sha256(JCS(record_without_edge_id))`, validate dependency closure/priority/source hashes and scheduler/batcher input completeness, replay twice byte-identically, and require the claim allowlist to reject real end-to-end language.
16. **Gated P4 only:** if dynamic rollout is approved and implemented, execute prepare/load/verify/CAS/drain/rollback/failure probes and require coverage/epoch/retention invariants. Until then every dynamic rollout metric and claim is `NA`.
17. **Document and each frozen run:** freeze the artifact hash for independent decision-drift, scope/simplicity, technical-correctness, and reflection/proof-boundary reviews.

## Appendix A. Requirements traceability

The following 40 rows correspond one-for-one and in source order to the 40 bullets in the original project brief. “Planned” means contracted for a named phase; it does not mean implemented.

| ID | Normative source bullet | Design/interface | Config | Event/metric | Validation | Phase/status |
|---|---|---|---|---|---|---|
| REQ-001 | MoE model: start with a manageable open-weight MoE such as Qwen3-30B-A3B. | §§1,4 `ModelRuntime` | `model-spec.v1` | manifest facts | pinned revision/loader probe | P0 planned |
| REQ-002 | Inference API: OpenAI-compatible endpoint for prompts, streaming generation, concurrency, and benchmark clients. | §§5,8 API adapter | API compatibility manifest, workload | request/stream events | endpoint/SSE/concurrency/cancel suite | P1 planned; exact subset gated |
| REQ-003 | Inference coordinator: owns each request/session, executes the shared Transformer path, invokes routers, dispatches selected experts, and combines expert outputs. | §§5,8 `ModelRuntime`/`RemoteMoeBlock` | shared strategy | route/dispatch/combine | end-to-end ownership probe | P1 planned |
| REQ-004 | Expert workers: independent processes capable of loading assigned (layer, expert) weights and executing batched expert forward passes. | §§5–9 `ExpertWorker` | placement, hardware, batching | residency/queue/compute | extracted-expert/batch probe | P0–P1 planned |
| REQ-005 | Expert directory: maps every (layer_id, expert_id) to one or more worker replicas and tracks availability. | §§5–6 `ExpertDirectory` | directory snapshot/placement | directory/residency/health | coverage and revision integrity | P1 planned |
| REQ-006 | Configurable expert placement: arbitrary assignment and replication of experts across workers without changing model code. | §§5,9 `PlacementPolicy` | `placement-policy-spec.v1`, `placement-plan.v2` | plan/objective/feasibility events | exhaustive arbitrary-map and hard-constraint probe | P1 manual; P2A policies; P2B replication |
| REQ-007 | Router instrumentation: record router scores and selected experts for every token and every MoE layer. | §§8,12 | output capture policy | `route_decided` full 128 scores | expected token-layer completeness | P1 planned |
| REQ-008 | Execution tracing: record request/token/layer/expert IDs, dispatch times, queue delay, compute time, response time, activation size, and worker ID. | §§6,12 telemetry | event schema | common envelope and typed spans | referential/temporal integrity | P1 planned |
| REQ-009 | Multiple physical nodes: initially a small heterogeneous set of Mac and/or Linux machines (NVIDIA CUDA, Apple Silicon/MPS, or CPU); large-scale consumer-hardware fleets can be added later. | §§10,18 | hardware/topology | node/resource events | physical-node inventory/probe | P1 planned (Mac and Linux, CUDA/MPS/CPU); large-scale fleet P4 deferred |
| REQ-010 | Logical regions: assign nodes labels such as us-west, us-east, europe, asia, independent of their real physical location. | §§10–11 | `logical_region_id`, `physical_site_id` | locality fields | label independence audit | P1 planned |
| REQ-011 | Network emulation: configurable RTT/one-way delay, bandwidth, jitter, packet loss, and connection failures between every pair of logical regions/nodes. | §§10–11 `TopologyController` | ordered `links`, faults | topology/fault/probe events | all-pairs actual-path probe | P1 planned |
| REQ-012 | Asymmetric network support: uplink/downlink and A→B/B→A characteristics must be independently configurable. | §§10–11 | node caps + two ordered links | directional counters/utilization | independent reverse-direction probe | P1 planned |
| REQ-013 | Topology configuration: define machines, regions, links, capacity, and hardware characteristics through a reproducible config file. | §11 | topology/hardware schemas | factor hashes | schema/hash recreation | P1 planned |
| REQ-014 | Hardware profiles: accelerator backend (CUDA/MPS/CPU) and model, memory capacity, compute throughput, memory bandwidth, and optionally measured expert execution performance. | §11 | `hardware-inventory.v1`, calibration | resource/service samples | declared-vs-measured audit | P0 planned; values gated |
| REQ-015 | Worker heterogeneity: different workers can have different compute speed, memory, network bandwidth, and expert capacity. | §§9–11 | per-node profile/capacity | queue/compute/link/memory | heterogeneous worker probe | P2 planned |
| REQ-016 | Replication support: the same expert can reside on multiple workers. | §§6,9 | placement replicas | residency/replication metrics | exact replica equivalence | P2 planned |
| REQ-017 | Runtime replica selection: scheduler can choose among replicas based on latency, queue length, compute speed, locality, reliability, or arbitrary experimental policies. | §§5,9 | scheduler spec/calibration | decision candidates/scores | policy and fallbacks probe | P2 planned |
| REQ-018 | Pluggable scheduling policy: clean interface for replacing expert-selection/placement/runtime-dispatch algorithms without modifying the inference engine. | §§5,7,9 `SchedulingPolicy` | `scheduler-spec.v1` | scheduler decisions | interface substitution/isolation | P1 interface; P2 alternatives |
| REQ-019 | Pluggable placement policy: separate offline/dynamic algorithm controlling which experts are stored or replicated on which machines. | §§5,7,9 `PlacementPolicy` | study snapshot, policy spec, `placement-plan.v2` | plan/feasibility/application events | pure-input, forbidden-field, determinism and isolation audit | P1 manual; P2 offline; dynamic P4 gated |
| REQ-020 | Batching: batch multiple tokens routed to the same expert/worker so the benchmark isn’t artificially handicapped by one-RPC-per-token execution. | §§7,9 | three-stage batching spec | logical batch/RPC/kernel IDs | one-row vs multi-row equivalence | P1 planned |
| REQ-021 | Efficient RPC transport: binary tensor transport rather than JSON/HTTP serialization for expert activations. | §7 `ExpertTransport` | payload/message limits | byte/staging/RPC events | raw-bit round trip | P0–P1 planned |
| REQ-022 | Backpressure and queues: model realistic worker contention instead of assuming every worker is always idle. | §§6,9 | queue/buffer/concurrency limits | admission/queue/resource events | saturate every bounded resource | P1 planned |
| REQ-023 | Node failure/churn simulation: workers can disappear, slow down, reconnect, or lose particular expert replicas. | §§6,10 | `fault-plan.v1` | fault/health/incarnation/attempt | full failure suite | P1 essentials; broader P4 |
| REQ-024 | Warm/cold expert state: distinguish weights already in RAM/VRAM from experts requiring loading or movement. | §§6,15 | residency/warm state | load/residency events | cold-load/unload/reload | P1 state; movement P4 |
| REQ-025 | Shared-weight handling: explicit strategy for attention/router/shared parameters—replicated, centralized, or distributed—so their cost isn’t accidentally ignored. | §§1,4–5 | `COORDINATOR_CENTRALIZED` | shared-path/critical-path events | exhaustive tensor/accounting probe | P0–P1 planned |
| REQ-026 | Correctness mode: distributed execution should exactly match the reference model when using the original router and weights, within numerical tolerance. | §§2,14 | mode/tolerance preregistration | tensor/logit/route/token checks | baseline matrix | P1 planned; tolerances gated |
| REQ-027 | Baseline implementation: normal single-node or datacenter-style inference using the same model and precision. | §14 | baseline manifest | comparison artifacts | identical-factor baseline probe | P0 planned; hardware gated |
| REQ-028 | Workload suite: interactive single-user decode, concurrent requests, short/long prompts, short/long generations, and representative text datasets. | §15 | `workload-spec.v1` | request/token outcomes | workload manifest coverage | P2 planned; dataset gated |
| REQ-029 | Repeatable routing traces: ability to capture real expert paths and replay exactly the same traces against different simulated network topologies and placement algorithms. | §§12,16 | capture/replay mode/calibration | route/dependency events | completeness + byte-identical replay twice | P3 deferred until LIVE_E2E |
| REQ-030 | Core latency metrics: TTFT, inter-token latency, tokens/sec/request, p50/p95/p99 latency. | §13.1 | metric schema | request/token timestamps | formula/sample-count audit | P1 planned |
| REQ-031 | MoE-specific metrics: expert accesses/token, slowest expert time per layer, expert queue time, load imbalance, replica utilization. | §13.2 | metric schema | route/queue/compute events | formula/denominator audit | P1 core; replica P2 |
| REQ-032 | Network metrics: bytes sent/token, RPCs/token, WAN RPCs/token, RTT contribution, bandwidth utilization. | §13.3 | topology/metric schema | byte/RPC/counter/probe events | packed-attempt and paired-shaping audit | P1 planned |
| REQ-033 | Locality metrics: percentage of expert accesses satisfied machine-local, LAN-local, region-local, and cross-region. | §13.4 | node/LAN/region fields | accepted/attempt locality | exclusive-class reconciliation | P1 planned |
| REQ-034 | Sequential synchronization metric: number of cross-machine and cross-region communication barriers per generated token. | §13.4 | retry-round semantics | dispatch dependencies/attempt rounds | DAG barrier derivation | P1 planned |
| REQ-035 | Capacity metrics: aggregate tokens/sec, requests/sec, worker utilization, and memory utilization. | §§13.1,13.5 | measurement interval/capacity | completion/compute/resource | interval and union audit | P1 planned |
| REQ-036 | Replication metrics: memory/storage overhead versus locality/latency improvement. | §13.5 | placement/residency | bytes/locality/latency | paired single-copy comparison | P2 planned |
| REQ-037 | Quality evaluation: mandatory once the router itself is modified; compare generated quality/benchmark accuracy against unmodified routing. | §§2,14 | forced/modified mode + quality plan | quality artifact | claim gate vs unmodified router | P4/deferred unless routing changes |
| REQ-038 | Experiment specification: model version, quantization, topology, expert mapping, scheduler, random seed, workload, and network conditions captured with every run. | §§3,11 | full factor bundle/manifest | run start/end/hashes | resolved-config recreation | P1 planned |
| REQ-039 | Result store: persist raw traces plus aggregate benchmark results so algorithms can be compared over time. | §17 `ResultStore` | output/retention policy | raw events + metrics | hashes/COMPLETE/immutability | P1 planned; retention gated |
| REQ-040 | Visualization: at minimum latency breakdown, expert-location/path visualization, worker utilization, expert popularity, cross-layer co-activation, and local-vs-WAN hit rates. | §§13.4,17 `ResultAnalyzer` | visualization/study schemas | named logical and proof-gated physical input tables | plot completeness plus physical-proof `NA` fixture | P2 planned; physical deployment gated |

The source's final separation sentence is enforced across §§3,5,7,9 and by the `vary_only` isolation validator: `ModelRuntime`, `PlacementPolicy`, `TopologyController`, `SchedulingPolicy`, and `BatchingPolicy` have distinct identities, state, interfaces, and forbidden ownership.

## Appendix B. Revision-4 acceptance ledger

This ledger supplements rather than duplicates the 40 normative source rows.

| Owner clause / semantic closure | Section | Artifact/event/metric | Validation | Phase |
|---|---|---|---|---|
| Directional application-path latency discovery with coverage/age/confidence and authority | §§10–12,17 | `latency-discovery-snapshot.v1`, latency probe events, discovery matrix | §23 item 8 fail-closed cells/authority | P2A |
| Grouped routing partitions and prefill/decode frequency/co-activation | §§9.1,11,17 | partition/statistics artifacts, route events | §23 item 9 recomputation/leakage audit | P2A |
| Sealed common planning root and test commitment | §§3,7,11,15 | `placement-study-snapshot.v1`, seal/unseal events | §23 items 9 and 13 access chronology | P2A/P2B |
| Pure policy inputs/outputs; no test/live state or online update | §§5,7,9.1,15 | policy spec, plan v2, feature audit events | §23 items 9, 11, and 13 | P2A/P2B |
| Normative directional objective; cross-layer descriptive only | §9.1 | objective spec/components | §23 item 10 symbol/input recomputation | P2A/P2B |
| Complete hard constraints and deterministic feasibility/ties | §§9.1,11 | feasibility proof, iteration events | §23 items 10–11 fixtures/repeat bytes | P2A/P2B |
| Plan excludes runtime replica order/queue/health/attempt/scheduling | §§5,7,9.1–9.2,11 | plan schema and scheduler decision events | §23 item 11 forbidden-field rejection | P1/P2 |
| Manual immutable startup correctness | §§6.2,11–12,19 | application record/state events/directory | §23 item 12 failure injection | P1/P2 |
| Exactly one copy in Phase 2A; common owner-approved budget only in Phase 2B | §§9.1,15.2,19 | plan/budget/feasibility fields | §23 items 11 and 13 | P2A/P2B |
| Six policy definitions and feature masks | §§9.1,15 | policy specs and roster | §23 item 13 | P2A |
| Frozen fair live/replay comparisons and no forced live routes | §§15–16 | comparison contract, route-equality predicate | §23 item 13; optional item 15 replay | P2/P3 |
| Placement quality, convergence, instability, churn, movement, readiness | §§12–13.5,17 | plan/iteration/startup events and named metrics | §23 items 10–13 derivation | P2 |
| Phase-aware campaign matrix and claim ceilings | §§2,15.2,19 | cell IDs, run/validation records | §23 items 13 and 15–16 | P1–P4 |
| All-phase and decode-only barriers, including prefill/one-token/retries/waits | §13.4 | dependency edges and dual metric families | §23 item 6 fixtures | P1/P2 |
| Physical-WAN/non-WAN visualization fails closed with counts | §§2,10,13.4,17; REQ-040 | topology proof, physical classes/rates | §23 item 14 and REQ-040 fixture | P2; deployment gated |
| Replay edge hashes record without its ID | §16 | `replay-dependency-edge.v1` | optional §23 item 15 self-hash rejection | P3 |
| Measured discovery, configured emulation, logical region, physical proof, and paired causal evidence remain distinct | §§2,10,12.2,13.3 | topology/discovery/proof/comparison artifacts | §23 items 5, 8, and 14 | P0–P4 |
| Snapshot freshness is admission, not live directory refresh | §§6.1–6.2,10 | discovery age predicate, immutable directory | §23 items 8 and 12 | P1/P2 |
| Replay is not a correctness prerequisite; dynamic rollout is documentation-only | §§6.2,16,19,22–23 | phase tags and claim matrix | §23 items 15–16 | P3/P4 gated |
| Hardware, deployment/security, API, tolerances, statistics/cells/budget, and physical validation remain owner gates | §§18,21–22 | resolved gate artifacts when approved | phase entry plus §23 evidence | Gated |

## Appendix C. Revision-5 semantic-closure ledger

The 20 revision-4 rows above remain unchanged in force. These focused rows close reproducibility gaps without adding a subsystem or selecting a Phase-2B budget.

| Closure | Section | Artifact/event/metric | Validation | Phase |
|---|---|---|---|---|
| One measured joint round-trip cost used once; all objective terms content-addressed and fail closed | §§9.1,10–12 | `objective-spec.v1`, `L_rpc`, keyed `S`, `objective_lookup` | §23 items 8 and 10 exact recomputation/error fixtures | P2A/P2B |
| Immutable candidate slots and one common complete constraint universe across policies | §§3,9.1,11,15.1 | `constraint-set.v1`, feasibility proof, comparison equality hashes | §23 items 10–11 and 13 slot/tier/mutation fixtures | P2A/P2B |
| Policy-independent canonical `FixedPrimary` and surrogate ordering | §§9.1–9.2,11 | `scheduler-spec.v1`, canonical replica identity/order | §23 item 11 row/insertion/serialization permutation fixture | P1/P2B |
| Full activation-statistic domains and typed zero-denominator `NA` semantics | §§9.1,11–12 | `activation-statistics.v1`, value/NA counts | §23 item 9 empty/absent/pair-domain/NA fixtures | P2A |

## Appendix D. Revision-6 semantic-closure ledger

The 20 revision-4 rows and four revision-5 rows above remain unchanged in force. This closure changes only the canonical projection used to compute their artifact identities.

| Closure | Section | Artifact/evidence | Validation | Phase |
|---|---|---|---|---|
| One non-self-referential identity rule and closed own-field registry for every structured artifact type | §§3,11–12.3,14.2,16–17 | schema-bundle registry, recomputed manifest/parent IDs, `ARTIFACTS.sha256`; specialized rules are aliases | §23 item 7 full positive/negative projection fixtures | P0–P3 |

## Appendix E. Revision-7 semantic-closure ledger

The 20 revision-4 rows, four revision-5 rows, and one revision-6 row above remain unchanged in force. This finite inventory closed the already-selected artifact graph without activating gated evidence or adding a service. Its `74`-name value below is an explicitly historical revision-7 audit snapshot, superseded by revision 8's exact current invariant of 77; the graph/type semantics remain active, but 74 is not an active count or acceptance condition.

| Closure | Section | Artifact/evidence | Validation | Phase |
|---|---|---|---|---|
| Exhaustive structured/raw/native/runtime identity inventory and recursively reachable file graph | §§3,11–12.3,16–17 | exactly 57 immediate structured types, 2 gated conditional contracts, 20 raw/native declarations, 5 runtime exclusions, historical revision-7 count of 74 classified names (superseded by current 77), six-column `ARTIFACTS.sha256` | §23 item 7 current counts, mutation, no-double-registration, and root-to-leaf reachability fixtures | P0–P3; quality/deletion gated |

## Appendix F. Revision-8 semantic-closure ledger

The revision-4 through revision-7 semantic requirements remain in force except that revision 8 explicitly supersedes revision 7's historical 74-name count with 77. This closure makes the existing six-column artifact index byte-exact and binds its aggregate references without adding storage, policy, or runtime behavior.

| Closure | Section | Artifact/evidence | Validation | Phase |
|---|---|---|---|---|
| Byte-exact index encoding, typed aggregate locators, exactly two roots, and non-composite native identities | §§2.1,11–12.3,17 | 77-name ledger; `sha256`/`input_hashes`/`factor_hashes` index bindings; canonical TSV/URI/parent/native-row rules | §23 item 7 aggregate, byte grammar, graph, native/retained-copy, and mutation fixtures | P0–P3 |

## Appendix G. Revision-9 semantic-closure ledger

All prior scientific, proof, abstraction, owner-gate, and MVP requirements remain active. This closure replaces four bare native-digest values with typed locators and clarifies historical count supersession without adding an identity name, artifact type, raw/native family, persistence choice, or security mechanism.

| Closure | Section | Artifact/evidence | Validation | Phase |
|---|---|---|---|---|
| Four typed native container locators, optional distinct retained-byte locators, and explicit supersession of historical 74 by current 77 | §§2.1,11–12.3,14.2,17 | validation analyzer, model runtime, run container fingerprint, and baseline analyzer locators; unchanged `57/2/20/5/77` closure | §23 item 7 four-locator resolution, rejection, independent mutation, reachability, and historical-count audit | P0–P3 |

## Appendix H. Revision-10 semantic-closure ledger

All revision-9 locator goals remain active, but its URI-only encoding is superseded by the digest-bearing locator below. This is a local child-sensitivity correction through ordinary parent JCS, not a new index identity, dereferenced hashing rule, type, family, service, persistence choice, or security mechanism.

| Closure | Section | Artifact/evidence | Validation | Phase |
|---|---|---|---|---|
| Exact index digest copied into all four native and optional retained locators, with stable-URI child and ancestor sensitivity | §§2.1,3,11–12.3,14.2,17 | four-member locator shapes, exact row equality, normative base/native/retained/order vectors; unchanged `57/2/20/5/77` and two roots | §23 item 7 all-four mutation paths, vector reproduction, mismatch/alias/composite rejection, and ancestor propagation | P0–P3 |

## Appendix I. Revision-11 semantic-closure ledger

Every earlier identity, locator, graph, scientific, proof, abstraction, owner-gate, and simplest-MVP requirement remains active. This closure publishes only complete synthetic preimages for the already-selected vectors; it does not select a real model revision or container, add an artifact field or type, or change the exact `57/2/20/5/77` closure and two-root graph.

| Closure | Section | Artifact/evidence | Validation | Phase |
|---|---|---|---|---|
| Complete canonical preimages and a noncanonical-order input for the existing child-sensitivity vectors | §§3,11.1 | three exact one-line full JSON/JCS byte sequences, one exact noncanonical input, `model-spec.v1` with `own_identity_field=none` | §23 item 7 independent hash recomputation, exactly-one-digest differences, parse-before-JCS permutation, and omission/extra/byte rejection | P0–P3 |
