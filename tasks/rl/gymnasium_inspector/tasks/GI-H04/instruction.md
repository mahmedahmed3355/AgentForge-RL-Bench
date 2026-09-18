# GI-H04 — Dependency Graph Completion

You are interacting with a Gymnasium environment whose task is to
complete every node in a dependency graph.

The public dependency graph is:

- Node 0: no prerequisites
- Node 1: no prerequisites
- Node 2: requires node 0
- Node 3: requires node 1
- Node 4: requires nodes 2 and 3
- Node 5: requires node 4

Your objective is to complete all six nodes while respecting every
dependency.

The environment uses six discrete actions:

- action 0 -> complete node 0
- action 1 -> complete node 1
- action 2 -> complete node 2
- action 3 -> complete node 3
- action 4 -> complete node 4
- action 5 -> complete node 5

A node cannot be completed before all of its prerequisites have been
completed.

A successful trajectory terminates when all six nodes are completed.

The environment follows the Gymnasium API:

reset() returns exactly:
    observation, info

step(action) returns exactly:
    observation, reward, terminated, truncated, info

The agent should reason about the complete dependency graph rather than
treating each node independently.

The verifier evaluates dependency closure and the resulting trajectory.
