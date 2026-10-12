"""Public method figure: the manuscript's Fig. 1 TikZ code, with the starting-pose drawing replaced by a
hand-specified schematic (tools/schematic_pose.py), compiled to PDF and converted to SVG.

    python tools/build_method_figure.py --manuscript ../manuscript/main.tex

The diagram text, layout and labels are taken verbatim from the manuscript; only the image in panel a
changes, so that the public page contains no pose derived from a recording.
Output: static/images/fig1_method.svg
"""
import argparse
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = HERE.parent
BUILD = HERE / "build"

PREAMBLE = r"""\documentclass[border=2pt]{standalone}
\usepackage{amsmath}
\usepackage{stix}
\usepackage{bm}
\usepackage[scaled=0.92]{helvet}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc,fit,backgrounds,shapes.misc}
\definecolor{manual}{HTML}{0072B2}
\definecolor{automatic}{HTML}{C26A00}
\begin{document}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manuscript", default=str(SITE.parent / "manuscript" / "main.tex"))
    args = ap.parse_args()
    tex = Path(args.manuscript).read_text()
    m = re.search(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", tex, flags=re.S)
    if not m:
        raise SystemExit("no tikzpicture found in the manuscript")
    pic = m.group(0)
    if "fig1a_pose.pdf" not in pic:
        raise SystemExit("expected the starting-pose image fig1a_pose.pdf in the manuscript figure")
    pic = pic.replace("fig1a_pose.pdf", "schematic_pose.pdf")
    BUILD.mkdir(exist_ok=True)
    subprocess.run(["python3", str(HERE / "schematic_pose.py")], check=True)
    (BUILD / "fig1_method.tex").write_text(PREAMBLE + pic + "\n\\end{document}\n")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "fig1_method.tex"], cwd=BUILD,
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["pdftocairo", "-svg", str(BUILD / "fig1_method.pdf"), str(SITE / "static/images/fig1_method.svg")], check=True)
    shutil.copy2(BUILD / "fig1_method.pdf", BUILD / "fig1_method_public.pdf")
    print("wrote static/images/fig1_method.svg")


if __name__ == "__main__":
    main()
