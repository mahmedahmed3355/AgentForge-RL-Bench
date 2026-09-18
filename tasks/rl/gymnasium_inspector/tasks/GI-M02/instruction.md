# GI-M02 — Gymnasium Inspector: Termination and Truncation Semantics

Repair the provided Gymnasium environment so that its episode termination semantics are correct.

The environment contains a deterministic task in which reaching the target state is a successful terminal condition, while exceeding the configured episode horizon is a truncation condition.

Your implementation must preserve the public Gymnasium API and correctly distinguish termination caused by task success from truncation caused by the episode time limit.

Required behavior:

1. reset() returns exactly (observation, info).
2. step(action) returns exactly:
   observation, reward, terminated, truncated, info
3. Successful completion sets terminated=True and truncated=False.
4. Exhausting the configured horizon without successful completion sets terminated=False and truncated=True.
5. terminated and truncated must never both be True.
6. Returned observations must belong to observation_space.
7. Accepted actions must belong to action_space.
8. reward must be a float.
9. info must be a dictionary.
10. The environment must remain deterministic when reset(seed=...) is used with the same seed.

Do not modify the tests, oracle, or verifier.

The verifier evaluates externally observable behavior rather than implementation details.
