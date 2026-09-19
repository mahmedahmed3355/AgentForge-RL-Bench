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

## V2 Decision-Density Requirements

The task must be solved through meaningful engineering decisions,
not by repeating inspection, benchmark, or validation actions.

Repeated inspection, benchmark, and validation calls do not by
themselves constitute progress.

At least two important engineering choices must have delayed
consequences: a choice may appear locally successful but later
produce a correctness, synchronization, transfer, performance,
or recovery consequence.

The agent must diagnose consequential failures and choose an
appropriate recovery strategy. Technically valid solution paths
may differ; the verifier evaluates the resulting engineering
outcome rather than requiring one artificial action sequence.

Early engineering choices must affect later available actions,
state transitions, failure modes, recovery options, or final
performance/correctness.

The target horizon is approximately 80–100 meaningful actions.
Artificial repetition must not be used to reach the horizon.


## CUDA-H01 V2 Engineering Decision Contract

The solution must be driven by engineering decisions rather than
repeated observation calls.

Repeated inspect, benchmark, and validation actions do not
constitute meaningful progress and must not be used to inflate
the trajectory.

Different engineering choices must lead to different subsequent
states, available actions, failure modes, recovery options, or
final outcomes.

At least two important choices must have delayed consequences.
A choice may appear locally successful and only later reveal a
performance, correctness, transfer, synchronization, or resource
consequence.

When a failure becomes observable, the agent must diagnose the
cause and select an appropriate recovery strategy. Valid
technical solution paths may differ.

Early choices must constrain or influence later optimization,
diagnosis, recovery, or configuration decisions.

The target trajectory is approximately 80–100 meaningful steps.
Artificial repetition must not be used to satisfy the horizon.


## Decision-Density Requirements

The solution must be driven by engineering decisions rather than
repeated inspection, benchmarking, or validation calls.

The agent should:

1. Establish the initial transfer and stream configuration.
2. Choose an appropriate transfer strategy based on the observed
   workload and synchronization constraints.
3. Choose a stream configuration that is compatible with the
   selected transfer strategy.
4. Validate the resulting configuration and interpret the outcome.
5. When a delayed consequence appears, diagnose its cause before
   changing configuration.
6. Choose a recovery strategy from technically valid alternatives,
   such as rollback, reconfiguration, or synchronization repair.
7. Revalidate after recovery and confirm the final correctness and
   performance state.

Repeated inspect, benchmark, and validate actions without changing
the engineering state should not be used to manufacture progress.

Important decisions must have downstream consequences. A transfer
strategy can affect stream behavior, and stream configuration can
affect synchronization and later validation. Therefore the agent
must reason from the current state instead of replaying a fixed
action sequence.

Multiple technically valid paths are allowed when they reach a
correct final state.

The target trajectory should remain approximately 80–100 meaningful
steps. Do not add no-op actions solely to increase trajectory length.


## V2 Decision-Density Contract

The task is evaluated on engineering outcomes and causal decisions.

Observation-only actions such as repeated `inspect`, `benchmark`, and
`validate_output` calls do not constitute meaningful progress.

Engineering decisions must alter subsequent state, available actions,
failure modes, recovery choices, or final outcomes.

At least two important decisions must have delayed consequences.
A locally successful configuration may later expose a correctness,
synchronization, transfer, or performance consequence.

When a consequential failure appears, the agent must diagnose it and
select an appropriate recovery strategy. Multiple technically valid
recovery paths remain acceptable where supported by the environment.

The oracle must use meaningful decisions rather than artificial
repetition. The verifier must evaluate the resulting state and
engineering outcome rather than requiring one fixed action sequence.

The target horizon is approximately 80–100 meaningful actions.

## CUDA-H01 V2 Decision-Density Contract

The task must be solved through meaningful engineering decisions.

Repeated inspection, benchmarking, comparison, and validation
must not be used as artificial progress or trajectory padding.

The agent must make consequential engineering choices involving
different CUDA transfer, stream, synchronization, performance,
correctness, or recovery strategies.

Different choices must produce different downstream states,
available actions, failure modes, recovery options, or outcomes.

At least two important decisions must have delayed consequences:
a decision that initially appears successful may later cause a
correctness, synchronization, transfer, or performance consequence.

When a consequential failure occurs, the agent must diagnose the
cause and choose an appropriate recovery, rollback, or
reconfiguration strategy.

Multiple technically valid solution paths are allowed where the
engineering outcome is equivalent.

Reward must come primarily from meaningful engineering progress
and final correctness/performance, not repeated observation calls.

The intended trajectory horizon is approximately 80–100
meaningful actions. Artificial repetition must not be used to
reach the horizon.


## CUDA-H01 Final Polish Contract

The task must be solved through meaningful engineering decisions.

Repeated observation-only actions such as `inspect`, `benchmark`,
`benchmark_stage`, `validate_output`, `compare_runs`, and
`inspect_errors` must not be used as artificial trajectory padding.

### Decision Density

Meaningful progress must come primarily from engineering decisions,
configuration changes, diagnosis, recovery, optimization, and
state-changing actions.

### Causal Branching

Engineering choices must have causal effects on subsequent state.

A different transfer strategy, stream configuration, synchronization
strategy, recovery strategy, or optimization choice must be capable
of changing later available actions, failure modes, performance,
correctness, or recovery requirements.

### Dependent Decisions

Earlier engineering choices must constrain or influence later
decisions. The agent should not be able to treat every decision as
independent of previous configuration.

### Delayed Consequences

At least two important engineering decisions must be capable of
appearing locally successful before producing a later consequence.

Relevant consequences include:

- transfer correctness,
- synchronization behavior,
- stream ordering,
- performance degradation,
- delayed failure,
- recovery requirements.

### Failure and Recovery

When a consequential failure occurs, the agent must diagnose the
failure and select an appropriate recovery path.

Valid recovery paths may include rollback, reconfiguration,
synchronization changes, transfer-strategy changes, or other
technically valid alternatives supported by the environment.

### Verifier Principle

Successful completion is determined primarily by the resulting
engineering state and outcome:

- correctness,
- performance,
- synchronization validity,
- recovery validity,
- final task state,
- and trajectory validity.

Calling a particular tool is not sufficient for success.

Multiple technically valid solution paths must remain acceptable.

### Oracle Principle

The oracle should demonstrate meaningful engineering decisions.
Observation calls must not be used merely to inflate the trajectory.

The target horizon remains approximately 80–100 meaningful actions.
Artificial padding is not a valid solution mechanism.
