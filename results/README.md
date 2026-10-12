# Aggregate results

Summary statistics behind the paper. Every file is aggregated over clips (per model, per test, per
caption type or per training run); no trajectories, poses or clip-level values are included.
Model identifiers are `<run>@<step>`; runs are named `v7_manual_s<seed>`, `v7_automatic_s<seed>` and
`abl_minus_{all,label,magnitude,balance}_s<seed>` (V7 is the model called NEST in the paper).

| folder | content | produced by |
|---|---|---|
| `test_partA/` | manual vs automatic captions and baselines on the locked test split: `commands.csv` (success per limb × test), `validate.csv` (reconstruction metrics), `by_caption_type.csv` (amplitude ratio and error by caption type), `kinematics.csv` (W1 and Fréchet distances), `partA.json/.md` (statistics and gate) | `analysis/pipeline.py analysis/configs/test_partA.yaml`, `analysis/stats.py partA` |
| `test_partB/` | the same for the 12 ablation models; `partB.json/.md` (effects vs full model) | `analysis/configs/test_partB.yaml`, `analysis/stats.py partB` |
| `validation_sweep/` | the same tables for every EMA snapshot of all 18 runs (training dynamics, checkpoint selection) | `analysis/configs/val_sweep.yaml` |
| `validation_selected/` | statistics of the selected checkpoints on the validation split | `analysis/stats.py ... --selection` |
| `calibration_diversity/` | calibration slope and correlations, sample spread per run (`per_run.csv`, `summary.csv`); pairwise spread per command for the single-infant case (`hero_apd.csv`) | `analysis/diversity_calibration.py` |
| `paraphrase/` | paraphrase robustness by set and parser outcome (`summary.csv/.json`) and by caption type (`change_by_type.csv`) | `analysis/paraphrase/analyze_paraphrase.py` |
| `stage2/` | Stage-2 follow-through statistics (`summary.json`) | `analysis/stage2/followthrough.py analyze` |

The checkpoint chosen for every run is in `analysis/selection/selection.json`.
