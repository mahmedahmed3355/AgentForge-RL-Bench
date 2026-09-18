# GI-M01 — Gymnasium Environment Contract Inspection

Task ID: GI-M01
Difficulty: medium
Domain: gymnasium_inspector

The agent must inspect and repair a small Gymnasium-compatible environment so that
its reset/step lifecycle, observation contract, reward accounting, termination
semantics, and deterministic seeded behavior satisfy the public task contract.

The task is intentionally terminal-driven. The agent interacts with the workspace
through shell commands and modifies only the candidate environment files.

The reference solution is deterministic.
The verifier evaluates observable behavior and does not expose hidden oracle data.
