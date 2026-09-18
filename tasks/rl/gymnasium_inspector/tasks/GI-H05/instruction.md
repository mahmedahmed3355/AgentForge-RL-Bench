# GI-H05 — Ambiguous Failure Diagnosis

Diagnose the active root failure from observable evidence and recover the environment.

The visible symptom is intentionally compatible with more than one possible failure mode.
Do not treat the symptom itself as the root cause.

Use the available inspection evidence before committing to a diagnosis.

Required successful sequence:

1. Inspect the environment.
2. Use the resulting diagnostic evidence.
3. Diagnose the active root failure.
4. Recover from the diagnosed root failure.
5. Finish only after recovery.

The environment must preserve the Gymnasium reset()/step() contract.

A successful terminal transition must return:

- terminated=True
- truncated=False

An unsuccessful trajectory that exhausts the horizon must return:

- terminated=False
- truncated=True

The verifier checks the complete trajectory and diagnosis/recovery state.
