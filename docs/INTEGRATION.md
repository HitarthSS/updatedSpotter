# Integration and arm definitions

`my_gate_v2.py` exposes `decide(state, observation, step, budget, kind, interventions)` and `remember(state, source)`, and a file-based CLI. `my_gate_v1.py` supplies the unchanged public-telemetry feature/gating helpers; it is a required module, not an alternative final candidate. The tests exercise bounded reuse, invalidation and mandatory verification without a model.

The two `integration/recovery_explore/judge_driver_final_*.sh` snapshots show the authors' and candidate action loops. They are historical integrations with pinned local policy/Qwen servers, real RoboCasa assets and runtime paths; do not execute them before preparing and reviewing that environment. Root controller code contains no API credential. The selected configuration uses Qwen full context; it is distinct from the paper's GPT arm.
