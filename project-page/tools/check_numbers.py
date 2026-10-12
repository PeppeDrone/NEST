"""Verify every number shown on the page against the manuscript's generated values.

    python tools/check_numbers.py [--numbers ../manuscript/generated/numbers.json]

Each quantitative value in index.html is wrapped as <span class="num" data-key="MACRO">value</span>, where
MACRO is the LaTeX macro the manuscript prints. The check fails if a key is unknown or the displayed value
differs from the manuscript (after normalising minus signs and LaTeX math wrappers).
"""
import argparse
import html
import json
import re
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]


def norm(v):
    v = html.unescape(str(v))
    v = v.replace("\\ensuremath{-}", "-").replace("−", "-").replace("$", "").strip()
    v = re.sub(r"^p\s*[<=]\s*", "", v)
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--numbers", default=str(SITE.parent / "manuscript" / "generated" / "numbers.json"))
    args = ap.parse_args()
    numbers = json.loads(Path(args.numbers).read_text())
    page = (SITE / "index.html").read_text()
    spans = re.findall(r'<span class="[^"]*\bnum\b[^"]*" data-key="([^"]+)">([^<]*)</span>', page)
    errors = []
    for key, shown in spans:
        if key not in numbers:
            errors.append(f"unknown key {key!r} (shown {shown!r})")
        elif norm(shown) != norm(numbers[key]):
            errors.append(f"{key}: page shows {shown!r}, manuscript has {numbers[key]!r}")
    print(f"checked {len(spans)} values ({len(set(k for k, _ in spans))} distinct keys)")
    for e in errors:
        print("  MISMATCH", e)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
