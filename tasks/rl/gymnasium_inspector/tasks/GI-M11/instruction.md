# Task: GI-M11

## Objective

Solve the Gymnasium/Inspector environment task and reach the verified target state.

## Domain

RL / Gymnasium / Inspector

## Difficulty

medium

## Primary capability

generalization

## Environment

The environment is stateful and interaction-driven.

The agent must use observations and available actions/tools to determine the correct sequence of operations.

The target state is not assumed to be directly visible.

## Constraints

- Do not assume hidden state.
- Do not bypass the environment.
- Do not modify verifier or oracle artifacts.
- Do not rely on a fixed reference trajectory.
- The final state must satisfy the verifier.
- Intermediate actions must remain valid environment interactions.

## Reasoning requirements

The task is evaluated over an interaction trajectory rather than a single answer.

The agent may need to:

1. Inspect the current state.
2. Select an appropriate action.
3. Observe the resulting transition.
4. Update its plan.
5. Recover from incorrect intermediate decisions when possible.
6. Reach the final target state.
7. Produce behavior accepted by the verifier.

## Hidden evaluation

The evaluator may use unseen parameterizations and adversarial cases.

Visible task information must not expose the complete reference trajectory.

## Reward

The environment exposes multiple reward components:

- progress
- correctness
- tests
- efficiency
- terminal
- penalties

The final reward is the sum represented by the environment's RewardBreakdown contract.

Reward must reflect correct behavior rather than shortcut exploitation.

## Success

Success requires verified final-state correctness and a valid interaction trajectory.
