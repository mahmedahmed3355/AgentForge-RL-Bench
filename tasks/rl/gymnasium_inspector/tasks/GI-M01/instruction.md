# GI-M01 — Inspect and Repair a Gymnasium Environment

You are given a small Gymnasium environment containing a contract violation.

Your objective is to inspect the environment, identify the behavioral defect, and
repair the candidate implementation.

The repaired environment must satisfy all of the following:

1. reset() returns exactly (observation, info).
2. step(action) returns exactly:
   observation, reward, terminated, truncated, info
3. Returned observations must belong to observation_space.
4. Actions accepted by step() must belong to action_space.
5. reward must be a float.
6. terminated and truncated must be booleans.
7. info must be a dictionary.
8. The environment must support deterministic seeded reset behavior.
9. Episode state must reset correctly between independent episodes.
10. The terminal condition must be reached only through the intended task progression.
11. The environment must not expose hidden verifier state.
12. The implementation must preserve the declared Gymnasium spaces.

Do not modify verifier files, oracle files, tests, or hidden evaluation material.

Use the terminal to inspect the candidate implementation and run the available tests.

The final environment must remain importable without depending on the AgentForge
repository.
