"""Parser check of the frozen paraphrase set (run after freezing; never used to edit paraphrases).

    $PY analysis/paraphrase/check_parser.py

For every paraphrase, the deterministic limb parser (neomotion.conditioning.parse_prompt) is applied
and its four limb tokens (known, moving, amplitude, speed per limb) are compared with the tokens of
the original caption. tokens_match=True paraphrases form the main analysis; the others are reported
separately with the token fields that changed.
Output: analysis/paraphrase/parser_check.csv, parser_summary.json
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from neomotion.conditioning import parse_prompt          # noqa: E402

LIMBS = ("left_arm", "right_arm", "left_leg", "right_leg")      # conditioning order
FIELDS = ("known", "moving", "amplitude", "speed")


def tokens(prompt):
    return [tuple(int(x) for x in code) for code in parse_prompt(prompt)[0]]


def verify_frozen():
    text = (HERE / "paraphrases_v1.json").read_text()
    expected = (HERE / "paraphrases_v1.sha256").read_text().split()[0]
    if hashlib.sha256(text.encode()).hexdigest() != expected:
        raise SystemExit("paraphrases_v1.json does not match its frozen SHA-256")
    return json.loads(text)


def main():
    payload = verify_frozen()
    rows = []
    for item in payload["items"]:
        ref = tokens(item["original"])
        for version in ("A", "B"):
            text = item[f"paraphrase_{version}"]
            got = tokens(text)
            changed = [f"{limb}.{field}:{a[k]}->{b[k]}" for limb, a, b in zip(LIMBS, ref, got)
                       for k, field in enumerate(FIELDS) if a[k] != b[k]]
            rows.append({"caption_id": item["caption_id"], "version": version, "caption_type_manual": item["caption_type_manual"],
                         "n_test_clips": item["n_test_clips"], "original": item["original"], "paraphrase": text,
                         "tokens_match": not changed, "changed": "; ".join(changed),
                         "original_tokens": json.dumps(ref), "paraphrase_tokens": json.dumps(got)})
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "parser_check.csv", index=False)
    by_type = df.groupby("caption_type_manual").tokens_match.mean().round(3).to_dict()
    clip_weighted = float((df.tokens_match * df.n_test_clips).sum() / df.n_test_clips.sum())
    field_changes = df.changed.str.split("; ").explode().dropna()
    field_changes = field_changes[field_changes != ""].str.split(":").str[0].str.split(".").str[1].value_counts().to_dict()
    summary = {"n_paraphrases": len(df), "n_match": int(df.tokens_match.sum()),
               "match_rate": float(df.tokens_match.mean()), "match_rate_clip_weighted": clip_weighted,
               "match_rate_by_caption_type": by_type, "match_rate_by_version": df.groupby("version").tokens_match.mean().round(3).to_dict(),
               "changed_fields": field_changes}
    (HERE / "parser_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(df[~df.tokens_match][["caption_id", "version", "paraphrase", "changed"]].to_string(index=False, max_colwidth=90))


if __name__ == "__main__":
    main()
