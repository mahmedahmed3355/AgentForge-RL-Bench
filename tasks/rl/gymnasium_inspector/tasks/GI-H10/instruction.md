# GI-H10 — Interrupted Episode Resume

Resume an interrupted episode from its latest valid persistent checkpoint.

The checkpoint contains committed progress. Do not replay transitions that were
already committed before the interruption.

Required behavior:

1. Inspect the checkpoint and current state.
2. Resume exactly from the latest valid checkpoint.
3. Continue only the work that remains incomplete.
4. Do not reset or replay already committed progress.
5. Finish only after the final state is valid.

The valid checkpoint contains progress `2`. A successful trajectory resumes at
progress `2`, performs the remaining two progress transitions, and then finishes.

A correct terminal state must have:

- progress = 4
- checkpoint_progress = 2
- resumed = true
- committed = true
- finished = true
- duplicate_transition = false
- stale_state = false
