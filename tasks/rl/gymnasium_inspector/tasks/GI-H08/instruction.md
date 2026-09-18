# GI-H08 — Action Efficient Completion

You must reach the valid terminal state while using actions efficiently.

The environment contains several valid paths with different action costs.

Required reasoning:

1. Inspect the available state and options.
2. Select a valid execution strategy.
3. Avoid unnecessary actions.
4. Reach the required progress level.
5. Finish only after the terminal conditions are satisfied.

The efficiency budget is limited.

An action that does not contribute to completion may consume budget without
providing progress.

The goal is not merely to finish. The trajectory must remain valid and
within the efficiency target.

Do not assume that the first locally valid action is the best choice.
