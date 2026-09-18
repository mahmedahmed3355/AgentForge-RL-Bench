# AgentForge-RL-Bench Architecture

## Core pipeline

Agent
→ Environment
→ Observation
→ Action / Tool Call
→ State Transition
→ Reward
→ Trajectory
→ Checkpoint
→ Held-out Evaluation
→ Transfer Analysis
→ Report

## Initial implementation

The first vertical slice targets Backend engineering.

The architecture must remain domain-independent so CUDA/GPU and Distributed
Training environments can be added without changing the core experiment
protocol.
