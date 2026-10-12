# partA statistics (analysis/artifacts/test_partA)

| contrast | estimate | 95% CI | p (sign-flip, 1-sided) | p (Holm) |
|---|---|---|---|---|
| manual - automatic, all 16 cells | +0.333 | [+0.282, +0.384] | 0.031 | – |
| manual - automatic, amplitude_order | +0.122 | [+0.058, +0.192] | 0.031 | 0.156 |
| manual - automatic, speed_order | +0.529 | [+0.347, +0.703] | 0.031 | 0.156 |
| manual - automatic, moves_more_than_still | +0.004 | [+0.000, +0.013] | 0.062 | 0.156 |
| manual - automatic, specificity | +0.678 | [+0.628, +0.719] | 0.031 | 0.156 |
| body_scale_error, automatic - manual (positive favours manual) | -0.016 | – | 0.750 | 0.750 |

Absolute mean success: {"v7_manual": 0.95, "v7_automatic": 0.616}
**Gate (primary estimate > 0 and CI excludes 0): PASSED**

With n patients the smallest attainable exact one-sided p is 1/2^n; the CI and the per-patient differences carry the evidence.
