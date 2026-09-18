# GI-H07 — Multi-Condition Terminal Verification

You are operating a Gymnasium environment in which terminal success depends
on several interacting conditions.

Do not treat an apparently successful condition as sufficient.

The environment exposes:
- current state conditions,
- validation state,
- dependency state,
- integrity state,
- inspection information.

Required terminal invariants:

1. `base_ready` must be true.
2. `dependency_ready` must be true.
3. `integrity_ok` must be true.
4. The resulting state must have been revalidated after any repair.

The important boundary is that `dependency_ready` and `integrity_ok` are
related. A superficial success observation before revalidation must not be
treated as terminal success.

Available actions:

- `inspect`
- `validate`
- `resolve`
- `finish`

A correct strategy is to inspect the state, validate the requirements,
resolve the failed dependency, revalidate the resulting state, and only then
finish.

The verifier independently recomputes terminal validity from the trajectory.
Do not rely on a stale validation result.
