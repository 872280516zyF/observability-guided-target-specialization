# Unequal visual observability guides target-specific specialization in image-based inverse inference: code release

This package contains the curated scripts used for the grouped outer-validation
observability analysis, inverse prediction, matched controls, input/augmentation
and component ablations, grouped forward interpretation, and the reported public
stress tests.

Publication-figure plotting and manuscript-layout scripts are intentionally not
included. Numerical source data for the reported figures and tables are provided
separately from this code repository.

## Data placement

Unzip the separate dataset package into this directory. The resulting layout
must contain `data/images3/after`, `data/images3/before`,
`data/images3/patterns`, and `data/images3/grouped_outer_cv_20260730`.

## Primary reproduction

Install `environment/requirements.txt`, then run `bash run_reproduction.sh` on
a CUDA-capable Linux environment. The complete five-fold, three-seed suite is
computationally intensive. Individual stages can be run through the shell
scripts in `scripts/`.

## Scope

- `scripts/run_5090_grouped_outer_cv_20260730.sh`: observability and inverse suite.
- `scripts/run_5090_grouped_forward_oof_20260730.sh`: grouped forward models.
- `scripts/run_5090_grouped_input_augmentation_ablation_20260731.sh`: input controls.
- `scripts/run_5090_textile_module_attribution_20260802.sh`: component controls.
- dated 20260804--20260807 scripts: subsequent exploratory refinements and public stress tests.

Model weights and cached outputs are intentionally excluded. Select a license,
complete `CITATION.cff`, and archive a complete dependency lock before making the
repository public.
