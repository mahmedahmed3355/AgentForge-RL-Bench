# AgentForge-RL-Bench
# Gymnasium / Inspector Task Specification v1

## 1. Task Identity

Every task MUST define:

- task_id
- version
- name
- domain
- family
- difficulty
- capabilities
- tags

Example:

task_id: GI-M01
version: 1.0.0
domain: gymnasium_inspector
family: state_reasoning
difficulty: medium

---

## 2. Difficulty

Only three difficulty levels are allowed:

- medium
- hard
- very_hard

Easy tasks are NOT part of this task family.

Target distribution:

- 11 Medium
- 11 Hard
- 11 Very Hard
- Total: 33

Difficulty MUST NOT be determined only by max_steps.

Difficulty may increase through:

- longer horizons
- branching decisions
- state dependencies
- delayed consequences
- hidden constraints
- recovery requirements
- multi-capability composition
- persistence
- generalization
- adversarial conditions

---

## 3. Capability Contract

Each task MUST explicitly declare the capability being tested.

Possible capabilities include:

- observation
- state reasoning
- planning
- action selection
- dependency tracking
- diagnosis
- recovery
- verification
- persistence
- optimization
- tool usage
- generalization

A task SHOULD have one primary capability and may have secondary capabilities.

---

## 4. TaskData

TaskData contains only agent-visible information.

Allowed:

- task_id
- version
- prompt
- scenario_id
- public parameters
- initial state parameters
- resource limits
- timeout
- seed
- public metadata

Forbidden:

- oracle solution
- hidden answers
- hidden tests
- verifier secrets
- evaluator secrets
- reference trajectory
- hidden scenario parameters
- gold solution

TaskData MUST be serializable and reproducible.

---

## 5. Environment Contract

The environment MUST expose a deterministic interaction contract.

Conceptually:

reset()
step(action)
close()

The environment MUST define:

- initial state
- observation
- action validation
- state transition
- reward
- termination
- truncation
- info
- reset semantics

The environment MUST support:

- deterministic reset
- seeded execution
- trajectory recording
- failure handling
- recovery where applicable
- terminal verification

---

## 6. State Model

State MUST be explicit.

Typical state fields:

- episode_step
- logical_stage
- current_phase
- completed_milestones
- objective_state
- validation_state
- performance_state
- failure_state
- failure_history
- recovery_count
- checkpoint_state
- terminal_state
- success_state

State transitions MUST be reproducible.

---

## 7. Logical Stages

Every task MUST define logical stages.

Example:

Stage 1
  Observe

Stage 2
  Inspect

Stage 3
  Diagnose

Stage 4
  Act

Stage 5
  Validate

Stage 6
  Recover

Stage 7
  Final Verify

Logical stages are NOT the same thing as agent actions.

One logical stage may require multiple actions.

---

## 8. Action Contract

Every action MUST define:

- action name
- arguments
- validation
- preconditions
- state transition
- postconditions
- reward effect
- failure behavior

Invalid actions MUST NOT produce unintended progress.

Actions MUST be observable in the trajectory.

---

## 9. Tools / Inspector

The Inspector layer is READ-ONLY from the agent's perspective.

Allowed:

- public state inspection
- public observations
- environment diagnostics
- public event history
- public progress information

Forbidden:

- hidden state
- verifier internals
- hidden tests
- oracle
- evaluator secrets
- hidden answers
- private filesystem secrets

Inspector output MUST represent information a real agent could legitimately obtain.

---

## 10. Objective

Every task MUST have:

- objective description
- success conditions
- terminal conditions
- failure conditions
- optional recovery conditions

The objective MUST be independently verifiable.

---

## 11. Reward

Reward MUST be component-based where appropriate.

Possible components:

- progress
- milestone
- correctness
- testing
- recovery
- performance
- efficiency
- terminal_success
- penalties

Terminal success MUST dominate shaping reward.

Reward MUST NOT encourage behavior that violates the actual objective.

---

## 12. Reward-Hacking Tests

Every task MUST test resistance against:

- reward farming
- repeated cheap actions
- reset farming
- duplicate submissions
- partial-credit loops
- timeout exploitation
- stale-state exploitation
- verifier manipulation
- hidden-file discovery
- benchmark manipulation

The genuine objective MUST be more valuable than reward exploitation.

---

## 13. Failure Modes

Every task MUST define expected failure classes.

Examples:

- invalid_action
- wrong_order
- wrong_state
- incorrect_configuration
- timeout
- failed_validation
- dependency_violation
- stale_state
- incomplete_solution
- recovery_failure

Failures SHOULD be observable and classifiable.

---

## 14. Recovery

Where recovery is part of the capability, the task MUST define:

- recoverable failures
- recovery actions
- recovery preconditions
- recovery cost
- recovery reward
- unrecoverable terminal states

Recovery MUST require meaningful reasoning.

---

## 15. Oracle

Every task MUST have an oracle/reference solution.

Oracle responsibilities:

1. Prove the task is solvable.
2. Produce a valid trajectory.
3. Reach a valid terminal state.
4. Produce valid final artifacts where applicable.
5. Pass the verifier.
6. Produce the expected reward range.

Oracle MUST NOT be exposed to the agent.

Oracle validation MUST happen before difficulty calibration.

---

## 16. Verifier

The verifier is the source of truth.

The verifier MUST independently validate:

- final state
- required artifacts
- correctness
- invariants
- terminal conditions
- hidden conditions
- performance requirements where applicable

The verifier MUST NOT trust an agent's self-reported success.

---

## 17. Hidden Evaluation

Tasks SHOULD support:

- train scenarios
- public evaluation scenarios
- hidden evaluation scenarios

Hidden evaluation MUST remain outside the agent-visible task data.

Hidden scenarios MUST be generated deterministically from controlled seeds.

---

## 18. Scenario Generation

Scenario generation MUST define:

- seed
- scenario_id
- parameter generation
- train distribution
- evaluation distribution
- hidden distribution
- reproducibility rules

Scenario generation MUST avoid train/evaluation leakage.

---

## 19. Determinism

Given:

- same task version
- same scenario
- same seed
- same action sequence

the environment SHOULD produce the same transition sequence.

This is required for:

- debugging
- replay
- benchmark validation
- trajectory analysis
- failure reproduction

---

## 20. Anti-Cheat

Every task MUST consider:

- filesystem inspection
- environment-variable inspection
- process inspection
- source inspection
- hidden-path discovery
- oracle discovery
- verifier discovery
- answer extraction
- reward manipulation

The task MUST expose only legitimate agent information.

---

## 21. Train / Eval Separation

Training data MUST NOT contain hidden evaluation scenarios.

Evaluation MUST be isolated from training.

Hidden evaluation MUST NOT be recoverable from public metadata.

---

## 22. Validation Gates

A task is NOT complete until:

1. package imports
2. config validates
3. TaskData validates
4. task generation works
5. environment reset works
6. environment step works
7. actions validate
8. reward validates
9. verifier validates
10. oracle passes
11. deterministic replay passes
12. reset isolation passes
13. anti-cheat tests pass
14. reward-hacking tests pass
15. failure/recovery tests pass
16. hidden evaluation tests pass
17. no-op baseline fails
18. trajectory is valid
19. documentation is complete

---

## 23. Definition of Done

A task is FROZEN only after all gates pass.

Oracle PASS
    ↓
Verifier PASS
    ↓
Anti-Cheat PASS
    ↓
Reward-Hacking PASS
    ↓
Determinism PASS
    ↓
Recovery PASS
    ↓
Difficulty Calibration
    ↓
Final Audit
    ↓
TASK FROZEN
