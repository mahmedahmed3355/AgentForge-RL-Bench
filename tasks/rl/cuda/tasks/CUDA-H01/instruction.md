# CUDA-H01 — Multi-Stage GPU Pipeline Recovery & Optimization

## Objective

Recover and optimize a stateful multi-stage GPU pipeline while satisfying both correctness and performance constraints.

Pipeline:

INPUT → H2D → PREPROCESS → TRANSFORM → REDUCTION → POSTPROCESS → D2H → VALIDATION → PERFORMANCE

The agent must investigate, form hypotheses, change configuration, execute experiments, diagnose failures, recover, re-plan, optimize, stress-test, and finally verify the complete pipeline.

## Important constraints

- The complete hidden bottleneck and failure configuration are not exposed.
- A fast but incorrect pipeline is invalid.
- A correct but underperforming pipeline is invalid.
- Intermediate reward does not imply success.
- Repeated inspection cannot substitute for engineering progress.
- Multiple valid optimization paths are accepted when they satisfy the verifier.

## Engineering concepts

The environment models:

- pageable vs pinned host memory
- H2D/D2H transfer strategy
- synchronous vs asynchronous execution
- CUDA streams
- CUDA events
- synchronization dependencies
- kernel blocks/shared memory/register pressure
- cross-stage dependencies
- workload variation
- delayed failures
- recovery and rollback
- deterministic performance scoring

## Delayed consequence

An asynchronous configuration can look locally successful and benchmark better, yet fail later under stress if the dependency is not synchronized correctly.

The agent must therefore reason about consequences that are not immediately visible.

## Execution modes

If CUDA/PyTorch CUDA is available, representative CUDA work is executed using CUDA tensors, streams, and events.

Without CUDA hardware, the same state/action/reward/verifier semantics are executed by a deterministic simulation model.

## Expected horizon

A successful reference trajectory is approximately 80 meaningful interactions, but the environment does not require an exact action count.

## Success

The final state must satisfy:

1. complete pipeline execution,
2. output correctness,
3. required performance target,
4. stress validation,
5. required recovery when the scenario contains delayed failure,
6. valid trajectory behavior,
7. no reward-hacking shortcut.

