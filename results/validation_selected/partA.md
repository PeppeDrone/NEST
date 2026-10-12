# partA statistics (analysis/artifacts/val)

| contrast | estimate | 95% CI | p (sign-flip, 1-sided) | p (Holm) |
|---|---|---|---|---|
| manual - automatic, all 16 cells | +0.297 | [+0.229, +0.361] | 0.062 | – |
| manual - automatic, amplitude_order | +0.089 | [-0.010, +0.185] | 0.125 | 0.375 |
| manual - automatic, speed_order | +0.461 | [+0.253, +0.659] | 0.062 | 0.312 |
| manual - automatic, moves_more_than_still | +0.013 | [+0.000, +0.039] | 0.125 | 0.375 |
| manual - automatic, specificity | +0.625 | [+0.544, +0.695] | 0.062 | 0.312 |
| body_scale_error, automatic - manual (positive favours manual) | +0.002 | – | 0.500 | 0.500 |

Absolute mean success: {"v7_manual": 0.944, "v7_automatic": 0.647}
**Gate (primary estimate > 0 and CI excludes 0): PASSED**

With n patients the smallest attainable exact one-sided p is 1/2^n; the CI and the per-patient differences carry the evidence.
