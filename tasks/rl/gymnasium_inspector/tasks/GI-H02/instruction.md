# GI-H02 — Interacting State Variables

Repair the provided Gymnasium environment so that its state remains
causally consistent across transitions.

The environment contains interacting state variables:

- `position`
- `energy`

The variables must be reasoned about jointly rather than independently.

Requirements:

1. `reset()` returns exactly `(observation, info)`.
2. `step(action)` returns exactly
   `(observation, reward, terminated, truncated, info)`.
3. Successful completion sets `terminated=True` and `truncated=False`.
4. Horizon exhaustion without success sets
   `terminated=False` and `truncated=True`.
5. `terminated` and `truncated` must never both be `True`.
6. Returned observations must belong to `observation_space`.
7. Accepted actions must belong to `action_space`.
8. `reward` must be a float.
9. `info` must be a dictionary.
10. The environment must remain deterministic when the same seed is
    supplied to `reset(seed=...)`.

The critical reasoning requirement is that `position` and `energy`
form a coupled state.  A transition is valid only when the joint state
invariant is preserved.

The environment should support inspection of the current state and
history through the information exposed by the environment.

The intended successful trajectory uses action `1` repeatedly until
completion.
