# GI-H06 — Progress Preserving Recovery

Recover from a failure without destroying valid previous progress.

Complete two milestones. A recoverable failure then becomes active. Use inspection,
recover while preserving the completed milestones, continue to the remaining
milestones, and finish only after all four milestones are complete.

Actions:
- 0 = inspect
- 1 = continue
- 2 = recover
- 3 = finish

A valid trajectory is:
inspect, continue, continue, recover, continue, continue, finish

The verifier checks the preservation invariant across recovery.
