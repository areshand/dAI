# dAI experiment record

This directory is the durable record of experiments for the decentralized-AI inference testbed.

- [`JOURNAL.md`](JOURNAL.md) is the chronological decision log. Add one entry for every paid run, local run that changes a decision, failed setup that reveals a constraint, or correction to an earlier claim.
- [`reports/`](reports/) contains one tracked report per experiment. Reports summarize the fixed contract, measured result, validity boundary, cost, cleanup, and next decision.
- [`REPORT_TEMPLATE.md`](REPORT_TEMPLATE.md) is the required structure for new reports.
- Large traces, server logs, and machine-readable outputs remain under `prototype/results/`. That directory is ignored by Git; every report must identify its run ID and raw-artifact path.

## Recording rules

1. Assign an ID in the form `EXP-YYYYMMDD-NN` before interpreting the result.
2. State the question and falsifiable hypothesis separately from the mechanism being tested.
3. Freeze the model, prompt, input/output lengths, cache policy, load, sampling, runtime, hardware, repetitions, and success gates.
4. Distinguish measured facts from interpretation. A loadability result is not a throughput result; output stability is not task-quality equivalence.
5. Preserve negative and failed runs. Mark their status and record what they ruled out.
6. Never silently rewrite history. Add a dated correction and link the superseding experiment.
7. Record resource cleanup for every cloud run. A run is not operationally complete until an independent inventory check finds no unintended live instances or volumes.

## Status vocabulary

- `validated`: the stated contract and gates passed.
- `negative`: execution succeeded and falsified the tested performance or quality hypothesis.
- `exploratory`: useful signal, but the sample size or proxy cannot support a final claim.
- `diagnostic`: isolates a mechanism or root cause rather than qualifying a product result.
- `failed-setup`: no target measurement, but the failure established an actionable constraint.
- `superseded`: a later experiment corrected the baseline, contract, or interpretation.
