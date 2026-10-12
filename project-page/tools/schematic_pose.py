"""Draw the schematic starting pose used in the public version of the method figure.

The coordinates below are hand-specified (a generic supine infant stick figure); they are NOT taken
from any recording, so the public figure contains no pose derived from clinical data.
Output: tools/build/schematic_pose.pdf
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "build"
# body-20 order: nose, neck, R sh, R el, R wr, L sh, L el, L wr, R hip, R knee, R ank, L hip, L knee, L ank,
# R eye, L eye, R ear, L ear, L foot, R foot   (image coordinates, y down; infant seen from above)
P = {0: (0.0, -1.55), 1: (0.0, -1.05), 2: (-0.42, -0.98), 3: (-0.78, -0.55), 4: (-0.70, -0.10),
     5: (0.42, -0.98), 6: (0.80, -0.62), 7: (0.95, -1.10), 8: (-0.24, 0.10), 9: (-0.55, 0.62),
     10: (-0.45, 1.15), 11: (0.24, 0.10), 12: (0.60, 0.55), 13: (0.40, 1.08), 14: (-0.10, -1.62),
     15: (0.10, -1.62), 16: (-0.20, -1.55), 17: (0.20, -1.55), 18: (0.48, 1.22), 19: (-0.55, 1.28)}
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12),
         (12, 13), (0, 14), (0, 15), (14, 16), (15, 17), (13, 18), (10, 19), (8, 11)]
LEFT = {5, 6, 7, 11, 12, 13, 15, 17, 18}
RIGHT = {2, 3, 4, 8, 9, 10, 14, 16, 19}


def main():
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(1.25, 1.25))
    for a, b in EDGES:
        c = "#004a75" if (a in LEFT or b in LEFT) else "#56B4E9" if (a in RIGHT or b in RIGHT) else "#3a3a3a"
        ax.plot([P[a][0], P[b][0]], [P[a][1], P[b][1]], color=c, lw=1.6, solid_capstyle="round")
    xs, ys = zip(*P.values())
    ax.scatter(xs, ys, s=5, color="#1a1a1a", zorder=3)
    ax.set_aspect("equal"); ax.axis("off"); ax.invert_yaxis()
    fig.savefig(OUT / "schematic_pose.pdf", transparent=True)


if __name__ == "__main__":
    main()
