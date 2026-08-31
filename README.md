# Unequal visual observability guides target-specific specialization in image-based inverse inference: code release

This repository contains the curated analysis and training code used for the
grouped outer-validation observability analysis, inverse prediction, matched
controls, input and component ablations, grouped forward interpretation, and
reported public-dataset stress tests.

## Data

The accompanying textile dataset is available from Zenodo:
https://doi.org/10.5281/zenodo.22169916

After downloading the data archive, extract it into the repository root so
that the following directories are available:

```text
data/images3/after
data/images3/before
data/images3/patterns
data/images3/grouped_outer_cv_20260730
```

Third-party public datasets used for stress tests are not redistributed here.
Please obtain them from their original sources cited in the article.

## Environment

Install the Python dependencies listed in `environment/requirements.txt` in a
CUDA-capable Linux environment. The reported experiments were run on an NVIDIA
GPU; full five-fold, three-seed reproduction is computationally intensive.

## Primary reproduction

Run:

```bash
bash run_reproduction.sh
```

Individual stages can be run through the shell scripts in `scripts/`:

- `run_5090_grouped_outer_cv_20260730.sh`: grouped observability analysis and inverse-model suite.
- `run_5090_grouped_forward_oof_20260730.sh`: grouped forward models.
- `run_5090_grouped_input_augmentation_ablation_20260731.sh`: input and augmentation controls.
- `run_5090_textile_module_attribution_20260802.sh`: component controls.
- dated 20260804-20260807 scripts: exploratory refinements and public-dataset stress tests.

## Repository scope

The repository includes the code required to reproduce the reported data
quality control, grouped splits, training-only observability ranking, inverse
and forward model training, matched controls, ablations, and statistical
summaries. Publication-figure layout scripts, manuscript files, model weights,
cached outputs, and third-party datasets are intentionally excluded.

## License

This software is released under the MIT License. See `LICENSE`.
