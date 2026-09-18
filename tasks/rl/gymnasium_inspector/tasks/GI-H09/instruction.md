# GI-H09 — Composed Inspector Queries

Use only the legitimate public Inspector operations.

Available operations:

- `inspect_state`
- `inspect_history`
- `inspect_progress`
- `inspect_diagnostics`

No single inspection is sufficient to reconstruct the complete state.

The required behavior is:

1. Inspect the available public information.
2. Correlate the complementary views.
3. Diagnose the resulting state.
4. Perform the correct action.
5. Verify the terminal state.

Do not infer the answer from one incomplete view.

Do not use hidden state, private attributes, filesystem shortcuts, or implementation details that are not part of the public Inspector contract.

The evaluator may redistribute information across the public diagnostic views while preserving the Inspector contract.
