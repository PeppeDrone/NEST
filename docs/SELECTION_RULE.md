# Checkpoint-selection rule (prespecified)

*Fixed on 6 October 2026, before the seed-43/44 and ablation runs were trained or evaluated.*
*Implemented in `analysis/select.py`; output `analysis/selection/selection.json`.*

**Candidates.** For each training run: every per-round EMA snapshot after the pose-only prior phase
(steps > 1,000, every 200 steps, until early stopping).

**Evidence.** Validation split only (115 clips, 4 patients). Paired command tests:
- 16 poses (deterministic round-robin over patients) × 2 sampling seeds × 13 commands;
- guidance 2.0, 50 DDIM steps.

This gives 16 cells: 4 limbs × {amplitude order, speed order, moves more than still, specificity}.
Each cell is the patient-mean fraction of successful pose/seed pairs.

**Rule.**
1. Choose the **earliest** step whose 16 cells are all ≥ 0.80.
2. If no step qualifies, choose the step that maximizes the minimum cell.
3. Ties go to the lower patient-mean body-scale error (validate stage, all 115 clips × 2 seeds), then to the earlier step.

**Scope.** The rule applies identically to:
- NEST manual and NEST automatic;
- every training seed;
- every Part-B ablation arm.

The test split is never used for selection.

**Disclosure.** Before this rule was written, seed-42 NEST manual results were inspected on validation, and step 3,200 was noted as satisfying all cells. The rule is applied mechanically to seed 42 as to all other runs, and the seed-42 choice is reported as whatever the rule returns.

**Numerical note.** Command tests are strict pairwise comparisons, so near-ties can flip with last-bit numerical changes. To keep them reproducible:
- each prompt's CLIP embedding is computed on its own (batch size 1);
- TF32 and fused attention kernels are disabled.

Recomputation is byte-identical (verified). Paired effect magnitudes (`commands/rows.csv`) are reported alongside the fractions.
