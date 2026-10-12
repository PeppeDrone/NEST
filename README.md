# NEST: controllable neonatal motion from a single frame and sparse movement specifications

Code, configurations, aggregate results and manuscript source for the paper
*“Controllable neonatal motion from a single frame and sparse movement specifications”*.

NEST generates 4.75 s of 2D infant motion (20 frames × 20 body keypoints) from one observed pose and a
short movement specification such as *“the newborn moves its left leg with high amplitude and fast
speed”*. It is a pose-anchored diffusion model with structured limb tokens, a per-limb magnitude term and
label-consistent kinematic bounds. The paper compares structured manual specifications with automatic
captions on identical data, analyses the model with loss ablations, paraphrase robustness and calibration,
and drives a frozen pose-conditioned video model with the generated trajectories.

## Repository layout

```
src/neomotion/        Stage 1: NEST model, conditioning, diffusion, training, inference
src/dmtbaby/          Stage 2: trajectory-to-video bridge, UniAnimate inference service, follow-through
configs/              v7.yaml (NEST) and the four ablation configurations
data_refs/            frozen patient split (pseudonymous infant IDs)
scripts/              environment, training launcher and queue
analysis/             evaluation pipeline, checkpoint selection, statistics, kinematics, timestep probe,
                      single-infant generation, calibration/diversity, paraphrase study, Stage-2 analysis
stage2/               screening of trajectories, input building, video inference; UniAnimate patch and pins
manuscript/           LaTeX source (Elsevier CAS), figure code, generated tables/numbers, figure PDFs
results/              aggregate results (CSV/JSON) behind every table and figure
docs/                 EXPERIMENTS.html (all experiments explained), DATA.md, SELECTION_RULE.md
```

## What is not included, and why

| item | status |
|---|---|
| Neonatal video recordings, reference stills, extracted poses, generated trajectories and videos | **not distributed** (clinical data of identifiable infants); format in `docs/DATA.md`; access subject to ethics approval, see the paper's Data availability statement |
| Trained checkpoints (18 selected models: manual, automatic, 4 ablations × 3 seeds) | released separately: [Google Drive](https://drive.google.com/drive/folders/1BlRaFzRa4hc5OnOmqcCp1vNBXxn5VZ5T) (`NEST_checkpoints_v1.zip`); layout below |
| UniAnimate-DiT code and Wan2.1-I2V-14B / adapter / DWPose weights | third-party; pinned commit, local patch and SHA-256 of every weight in `stage2/third_party/` |
| CLIP ViT-L/14 text encoder | downloaded from Hugging Face at the pinned revision (`openai/clip-vit-large-patch14@32bd642`) |

## Installation

```bash
python3.11 -m venv .venv && source .venv/bin/activate     # or a conda environment
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
export PY=python                                            # scripts/env.sh reads PY
```

All results were produced on an NVIDIA GB10 (aarch64, CUDA 13, 128 GB unified memory). Stage 1 also runs
on CPU (slowly); Stage 2 requires a large GPU.

For Stage 2:

```bash
git clone https://github.com/ali-vilab/UniAnimate-DiT.git external/UniAnimate-DiT
cd external/UniAnimate-DiT && git checkout 61d882c25385042f0cf5bcdaf6853238d9756d68 \
  && git apply ../../stage2/third_party/unianimate-61d882c.patch && pip install -e . && cd ../..
# place Wan2.1-I2V-14B-720P, the UniAnimate LoRA and the DWPose ONNX models as listed in
# stage2/third_party/model_provenance.json (paths relative to external/UniAnimate-DiT)
```

## Reproducing the paper

Every command runs from the repository root after `source scripts/env.sh`. Data must be laid out as in
`docs/DATA.md`. Steps 1–3 need the data; step 7 compiles the manuscript from the shipped figures and
tables without any data.

**1. Train** (18 runs; skips finished runs, resumes interrupted ones)
```bash
bash scripts/train_queue.sh 3            # all jobs in scripts/jobs_training.tsv, 3 in parallel
bash scripts/train_one.sh v7_manual_s42 configs/v7.yaml manual 42     # a single run
```
Every run writes `runs/<run>/` with an EMA snapshot at each validation round.

**2. Validation sweep and checkpoint selection** (rule fixed before the replicate runs: `docs/SELECTION_RULE.md`)
```bash
$PY analysis/pipeline.py analysis/configs/val_sweep.yaml   # every snapshot of every run (cached)
$PY analysis/select.py                                     # -> analysis/selection/selection.json
```
To use the released checkpoints instead, place them as `runs/<run>/ema_snapshots/step_<NNNNNN>.pt`.

**3. Locked test evaluation and statistics**
```bash
$PY analysis/pipeline.py analysis/configs/test_partA.yaml  # manual vs automatic + baselines
$PY analysis/stats.py partA analysis/artifacts/test_partA  # primary contrast and gate
$PY analysis/pipeline.py analysis/configs/test_partB.yaml  # ablations (run only if the gate passed)
$PY analysis/stats.py partB analysis/artifacts/test_partB --reference analysis/artifacts/test_partA
```

**4. Mechanism, guidance and single-infant analyses**
```bash
$PY analysis/probe_timesteps.py                            # fixed-noise timestep probe (validation)
for g in 1.0 1.5 2.5 3.0; do $PY analysis/pipeline.py analysis/configs/val_guidance_cfg$g.yaml; done
$PY analysis/hero.py                                       # rule-selected infant, 13 commands x 8 seeds
$PY analysis/diversity_calibration.py                      # calibration and sample diversity (CPU)
```

**5. Paraphrase robustness** (CPU, about 25 min)
```bash
$PY analysis/paraphrase/check_parser.py                    # frozen paraphrases_v1.json -> tokens kept/changed
CUDA_VISIBLE_DEVICES="" $PY analysis/paraphrase/run_paraphrase.py
$PY analysis/paraphrase/analyze_paraphrase.py
```
`analysis/paraphrase/build_paraphrase_file.py` documents how the frozen paraphrase set was written (by
Claude Opus 5.5, under fixed rules) and verifies its checksum; it never overwrites the frozen file.

**6. Trajectory to video (Stage 2)**
```bash
$PY stage2/screen.py                                       # bridge-readiness screen -> stage2/shortlist.csv
bash stage2/run_inference.sh                               # bridge + UniAnimate, 24 videos (~1 h 45 min each)
bash analysis/stage2/update.sh                             # DWPose follow-through, statistics, figures, PDF
```

**7. Manuscript**
```bash
bash manuscript/compile.sh      # regenerates figures/tables/numbers if analysis outputs exist, then pdflatex + bibtex
```
Every number in the text is a macro in `manuscript/generated/numbers.tex`; nothing is typed by hand.

### Figures and tables

| paper | file | code |
|---|---|---|
| Fig. 1 | `manuscript/main.tex` (TikZ), `figures/fig1a_pose.pdf` | `make_figures.py: fig1a` |
| Fig. 2 command following | `figures/fig2_command_following.pdf` | `fig2` |
| Fig. 3 caption types | `figures/fig3_caption_types.pdf` | `fig3` |
| Fig. 4 one infant, 13 commands | `figures/fig4_single_infant.pdf` | `fig4` |
| Fig. 5 original captions | `figures/fig5_original_captions.pdf` | `fig5` |
| Fig. 6 paraphrases | `figures/fig9_paraphrase.pdf` | `fig9` |
| Fig. 7 ablations | `figures/fig7_ablations.pdf` | `fig7` |
| Fig. 8 motion amount and reconstruction | `figures/fig6_stillness_trap.pdf` | `fig6` |
| Fig. 9 training dynamics and guidance | `figures/fig8_dynamics_guidance.pdf` | `fig8` |
| Fig. 10 trajectory to video | `figures/fig10_stage2.pdf` | `fig10` |
| Tables 1–4 | `generated/table_{cohort,partA,paraphrase,partB}.tex` | `tables`, `fig9` |

(File names follow the order in which figures were created; LaTeX numbers them by placement.)

## Released checkpoints

```
<run>/ema_step_<NNNNNN>.pt     selected EMA checkpoint (format dmtbaby.stage1.v7)
<run>/resolved_config.json     exact training configuration
<run>/training_summary.json    steps, stopping, best scores
<run>/metrics.jsonl            training and validation curves
selection.json                 selected step per run
SHA256SUMS
```
Checkpoints embed provenance (configuration, data and split checksums, pseudonymous clip identifiers of
the training/validation sets, summary validation metrics); they contain no images, poses or captions.

## Citation

```
<citation to be added on publication>
```

## License

Code: MIT (see `LICENSE`). Third-party components keep their own licenses (UniAnimate-DiT, Wan2.1,
DWPose, CLIP); the Elsevier CAS class files in `manuscript/` are distributed under the LaTeX Project
Public License.
