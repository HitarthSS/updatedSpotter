# My harness for Spotter — bounded supervision reuse

Experimental, frozen-policy harness code for the [paper](https://arxiv.org/abs/2609.36808). The evaluated behavior is in [`my_gate_v2.py`](my_gate_v2.py). Original authorship and source are credited in [NOTICE.md](NOTICE.md).

## Results

| Reference | Result |

| --- | --- |

| Paper / official release | 68.4%: pi0.5 + Qwen, full context |

| Our run of the authors' harness | 80/120 (66.67%) |

| **My harness** | **83/120 (69.17%)** |


Twenty-four reported RoboCasa tasks, five held-out-object split-B starts each; 240 physical episodes.


These are reduced or explicitly qualified experiments. They do not establish full-paper reproduction or a confirmed benchmark win. See [evaluation details](EVALUATION.md) and the compact records in [results/](results/).

## What my harness changes

1. Use corrected public gripper telemetry and a watchdog for stalled pre-grasp motion.

2. Reuse recent explicit OK judgements only under bounded image, pose and gripper change with a fixed expiration.

3. Invalidate reuse after contact changes, repairs, uncertainty or near the horizon; retain required post-repair verification.


Categories: `observation_grounding`, `execution_monitoring`, `uncertainty_compute`.

## Models and inputs

- Frozen DAVIAN-Robotics/pi05-robocasa-H50 policy and local Qwen3.8-27B-FP8.

- Qwen full-context configuration, released general lessons, no one-shot example and no paid GPT inference.


Policy weights remain frozen. Model checkpoints and benchmark assets must be obtained from their original releases under their own access terms. No paid LLM/VLM API is required by the published variant.

## Start here

1. Use a Linux GPU host for simulator/model evaluation. Read the [pinned upstream setup](https://github.com/zqc3117/Spotter) before installing its model stack.
2. Controller dependency: Pillow>=10,<12
3. Inspect `configs/method.json` and `provenance/upstream.json`.
4. Verify the source-only export and inspect the core API:

```sh
python3 tools/verify_export.py
python -m unittest test_my_gate_v2 -v
```


CPU checks validate code behavior or packaging only. They do not run the robot benchmark or demonstrate numerical parity on a new GPU/runtime.

## Integrate with the frozen policy

Read [the integration contract and arm definitions](docs/INTEGRATION.md).

The core gate exposes decide(state, public_observation, step=..., budget=..., kind=..., interventions=...) and remember(state, source). Use the CLI in my_gate_v2.py for the same file-based integration. integration/ contains the frozen author/my driver snapshots.


The core harness is reusable. Integration snapshots and historical experiment drivers are included for inspection; full fresh-machine simulator reproduction has not been revalidated during this publication. Prepare the pinned native environment and required assets before using those drivers. No benchmark is automatically launched by cloning, importing, or verifying this repository.

## Evaluation budget

The paper explicitly permits 1.8x each official task limit. Every recovery/calibration action counts.


Use identical tasks, initial states, seeds, models and budgets for the authors' harness and my harness. Charge every executed recovery/intervention to the action limit. Separate development and final cases, freeze the design before evaluation, and report all completed matched cases.

## Repository map

- `my_gate_v2.py`: main harness behavior.
- `configs/`: method and evaluation metadata.
- `results/`: compact measured outcomes and their interpretation.
- `provenance/`: source/model pins and export hashes.
- `tools/verify_export.py`: offline source integrity check.
- `reference/`, `integration/`, or `experiments/`, when present: native-loop snapshots.

## Known limitations

- Five helped and two harmed; exact McNemar p=0.4531; the success-difference interval includes zero.

- Calls increased from 1,646 to 1,679. Missing usage records prevent a complete token-cost comparison.

- A6000 FP8 kernels, dependencies and release revision differ from the historical paper environment.

- A retrospective 1.5x cutoff is not a separate experiment with a 1.5x planning budget.


Shared-GPU wall-clock measurements are descriptive. New checkpoints, a new serving runtime, different seeds or changed horizon enforcement require a newly labeled comparison.

## Publication checks

8 CPU tests passed. All exported Python parses, shell scripts pass `bash -n`, relative documentation links resolve, and source files pass a credential-pattern scan. See [validation](provenance/validation.json). A complete GPU benchmark was not rerun for this publication.
