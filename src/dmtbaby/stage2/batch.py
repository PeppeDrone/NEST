"""JSONL batch runner for the strict UniAnimate cascade service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .service import UniAnimateBatchService, UniAnimateModelConfig, UniAnimateRequest


def load_requests(path: str | Path, root: str | Path = ".") -> list[UniAnimateRequest]:
    manifest = Path(path).expanduser().resolve()
    requests: list[UniAnimateRequest] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {manifest}:{line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object on {manifest}:{line_number}")
            try:
                requests.append(UniAnimateRequest.from_mapping(row, root=root))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid request on {manifest}:{line_number}: {exc}") from exc
    if not requests:
        raise ValueError(f"Batch manifest contains no requests: {manifest}")
    outputs = [Path(request.output) for request in requests]
    duplicate_outputs = sorted({str(path) for path in outputs if outputs.count(path) > 1})
    if duplicate_outputs:
        raise ValueError("Batch manifest has duplicate output paths: " + ", ".join(duplicate_outputs))
    return requests


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run many strict 81-frame UniAnimate cascade requests while loading Wan once.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, help="JSONL request manifest.")
    parser.add_argument("--root", default=".", help="Base for relative request and model paths.")
    parser.add_argument("--unianimate_dir", default="external/UniAnimate-DiT")
    parser.add_argument("--wan_dir", default="external/UniAnimate-DiT/Wan2.1-I2V-14B-720P")
    parser.add_argument(
        "--lora_path",
        default="external/UniAnimate-DiT/models/UniAnimate-DiT/UniAnimate-Wan2.1-14B-Lora-12000.ckpt",
    )
    parser.add_argument(
        "--official_lora_path",
        default="external/UniAnimate-DiT/models/UniAnimate-DiT/UniAnimate-Wan2.1-14B-Lora-12000.ckpt",
        help="Official initializer loaded before a compact adapted checkpoint.",
    )
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--persistent_dit_params", type=int, default=6_000_000_000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def _resolve(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    requests = load_requests(_resolve(args.manifest, root), root=root)
    config = UniAnimateModelConfig(
        unianimate_dir=_resolve(args.unianimate_dir, root),
        wan_dir=_resolve(args.wan_dir, root),
        lora_path=_resolve(args.lora_path, root),
        official_lora_path=_resolve(args.official_lora_path, root),
        lora_alpha=args.lora_alpha,
        persistent_dit_params=args.persistent_dit_params,
        device=args.device,
    )
    outputs = UniAnimateBatchService(config).run_batch(requests)
    for output in outputs:
        print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
