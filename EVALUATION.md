# Evaluation: My harness for Spotter — bounded supervision reuse

Twenty-four reported RoboCasa tasks, five held-out-object split-B starts each; 240 physical episodes.

## Budget

The paper explicitly permits 1.8x each official task limit. Every recovery/calibration action counts.

## Interpretation

- Five helped and two harmed; exact McNemar p=0.4531; the success-difference interval includes zero.

- Calls increased from 1,646 to 1,679. Missing usage records prevent a complete token-cost comparison.

- A6000 FP8 kernels, dependencies and release revision differ from the historical paper environment.

- A retrospective 1.5x cutoff is not a separate experiment with a 1.5x planning budget.

## Reproduction discipline

Use the reported condition only. Establish native-path parity, prepare upstream models/assets and declare case manifests before expensive runs. This export itself is a source publication, not a new evaluation. Keep all historical and new cohorts separate.
