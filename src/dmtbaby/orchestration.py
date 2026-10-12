"""Separate-process cascade orchestration and strict Stage-2 resume handling."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .io import (
    atomic_write_json,
    atomic_write_jsonl,
    fingerprint,
    read_json,
    sha256_file,
    stable_seed,
)
from .manifests import RunStore, source_fingerprint
from .pose.schema import SCHEMA_ID
from .stage2.batch import load_requests
from .stage2.service import (
    DEFAULT_NEGATIVE_PROMPT,
    UniAnimateBatchService,
    UniAnimateModelConfig,
    UniAnimateRequest,
    adapter_checkpoint_chain,
    validate_request_images,
    validate_model_artifacts,
)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            clip_id = str(value.get("clip_id", "")).strip()
            if not clip_id:
                raise ValueError(f"Missing clip_id at {path}:{line_number}")
            value = dict(value)
            value["clip_id"] = clip_id
            rows.append(value)
    ids = [str(row["clip_id"]) for row in rows]
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise ValueError(f"Duplicate clip_id values in {path}: {duplicates}")
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def _control_hash(request: UniAnimateRequest) -> str:
    controls = request.validate_artifacts()
    return fingerprint(
        {
            "frames": [
                (path.name, sha256_file(path)) for path in controls.frames
            ],
            "reference_pose": sha256_file(controls.reference_pose),
            "reference_image": sha256_file(request.reference_image),
        }
    )


def _log_stage2(message: str) -> None:
    print(f"[stage2] {message}", flush=True)


def _model_provenance(
    root: Path,
    config: UniAnimateModelConfig,
    artifacts: Iterable[Path] | None = None,
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    provenance_path = root / "reproducibility/source-provenance.json"
    declared: Mapping[str, Any] = {}
    if provenance_path.is_file():
        payload = read_json(provenance_path)
        declared = payload.get("models", {}) if isinstance(payload, dict) else {}
    required = tuple(artifacts or validate_model_artifacts(config))
    actual_hashes: dict[str, str] = {}
    for index, path in enumerate(required, start=1):
        if verbose:
            size_gib = path.stat().st_size / (1024**3)
            _log_stage2(
                f"hashing model artifact {index}/{len(required)}: "
                f"{path.name} ({size_gib:.2f} GiB)"
            )
        actual_hashes[str(path)] = sha256_file(path)
    mismatches = {
        path.name: (declared[path.name], actual_hashes[str(path)])
        for path in required
        if path.name in declared and declared[path.name] != actual_hashes[str(path)]
    }
    if mismatches:
        raise ValueError(f"Stage-2 model artifacts disagree with declared SHA-256: {mismatches}")
    chain = adapter_checkpoint_chain(config)
    return {
        "official_adapter_path": str(config.official_lora_path),
        "selected_adapter_path": str(config.lora_path),
        "adapter_chain": [
            {"path": str(path), "sha256": actual_hashes[str(path)]}
            for path in chain
        ],
        "wan_dir": str(config.wan_dir),
        "declared_model_hashes": dict(declared),
        "actual_required_artifact_hashes": actual_hashes,
    }


def _stage2_status_path(output: Path) -> Path:
    return output.with_name(f"{output.name}.status.json")


def _read_status(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = read_json(path)
    return value if isinstance(value, dict) else None


def _archive_output(output: Path) -> None:
    """Move an explicitly overwritten result to a recoverable sibling tree."""

    status = _stage2_status_path(output)
    present = [path for path in (output, status) if path.exists()]
    if not present:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    archive = output.parent / ".replaced" / stamp
    archive.mkdir(parents=True, exist_ok=False)
    for path in present:
        os.replace(path, archive / path.name)


def _stage2_fingerprint(
    request: UniAnimateRequest,
    model_provenance: Mapping[str, Any],
    code_hash: str,
) -> str:
    settings = {
        key: value
        for key, value in asdict(request).items()
        if key not in {"output"}
    }
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in settings.items()}
    return fingerprint(
        {
            "config": settings,
            "code": code_hash,
            "checkpoint": model_provenance,
            "schema": SCHEMA_ID,
            "inputs": _control_hash(request),
        }
    )


def infer_stage2(args: Any, root: Path) -> int:
    """Run one validated batch while loading Wan/UniAnimate exactly once."""

    manifest_value = getattr(args, "manifest", None)
    control_value = getattr(args, "control_dir", None)
    legacy_pose_value = getattr(args, "pose_dir", None)
    direct_control = control_value or legacy_pose_value
    if manifest_value and direct_control:
        raise ValueError("--manifest cannot be combined with --control-dir/--pose-dir")
    if not manifest_value and not direct_control:
        raise ValueError("provide --manifest or direct --control-dir (legacy alias: --pose-dir)")
    if manifest_value:
        manifest = _resolve(root, manifest_value)
        requests = load_requests(manifest, root=root)
    else:
        reference_image = getattr(args, "reference_image", None)
        output = getattr(args, "output", None)
        if not reference_image or not output:
            raise ValueError("direct Stage-2 mode requires --reference-image and --output")
        reference_pose_value = getattr(args, "reference_pose", None)
        negative_prompt = getattr(args, "negative_prompt", None)
        requests = [
            UniAnimateRequest(
                clip_id=str(getattr(args, "clip_id", "direct")),
                control_dir=_resolve(root, direct_control),
                reference_pose=(
                    _resolve(root, reference_pose_value)
                    if reference_pose_value
                    else None
                ),
                reference_image=_resolve(root, reference_image),
                output=_resolve(root, output),
                prompt=str(getattr(args, "prompt", "")),
                negative_prompt=(
                    str(negative_prompt)
                    if negative_prompt is not None
                    else DEFAULT_NEGATIVE_PROMPT
                ),
                seed=int(getattr(args, "seed", 0)),
                height=int(getattr(args, "height", 832)),
                width=int(getattr(args, "width", 480)),
                num_frames=81,
                num_inference_steps=int(getattr(args, "num_inference_steps", 25)),
                cfg_scale=float(getattr(args, "cfg_scale", 1.5)),
                fps=int(getattr(args, "fps", 15)),
                quality=int(getattr(args, "quality", 5)),
                use_teacache=bool(getattr(args, "use_teacache", False)),
            )
        ]
    if not args.dry_run:
        from .io import require_free_disk

        for output_parent in {Path(request.output).parent for request in requests}:
            require_free_disk(
                output_parent,
                100.0,
                operation="production Stage-2 inference",
            )
    print(
        f"[stage2] validating images for {len(requests)} request(s)",
        flush=True,
    )
    for index, request in enumerate(requests, start=1):
        if index == 1 or index == len(requests) or index % 50 == 0:
            print(
                f"[stage2] image validation {index}/{len(requests)}: {request.clip_id}",
                flush=True,
            )
        validate_request_images(request)
    print("[stage2] image validation complete", flush=True)
    _log_stage2("resolving UniAnimate/Wan paths")
    official_adapter = _resolve(
        root,
        getattr(
            args,
            "official_adapter",
            "external/UniAnimate-DiT/models/UniAnimate-DiT/UniAnimate-Wan2.1-14B-Lora-12000.ckpt",
        ),
    )
    selected_value = getattr(args, "adapter", None)
    selected_adapter = (
        _resolve(root, selected_value) if selected_value else official_adapter
    )
    config = UniAnimateModelConfig(
        unianimate_dir=_resolve(root, args.unianimate_dir),
        wan_dir=_resolve(root, args.wan_dir),
        lora_path=selected_adapter,
        official_lora_path=official_adapter,
        persistent_dit_params=int(getattr(args, "persistent_dit_params", 6_000_000_000)),
        device=str(args.device),
    )
    _log_stage2("checking model artifact inventory")
    artifacts = validate_model_artifacts(config)
    _log_stage2(f"found {len(artifacts)} model artifact(s)")
    if selected_adapter != official_adapter:
        from .stage2.adapter_training import validate_training_adapter_checkpoint

        validate_training_adapter_checkpoint(selected_adapter)
    _log_stage2(
        "hashing model artifacts for resume fingerprint "
        "(multi-GB weights; often 2-10 min with no other output)"
    )
    model_provenance = _model_provenance(root, config, artifacts, verbose=True)
    _log_stage2("model artifact hashing complete")
    _log_stage2("hashing source tree for resume fingerprint")
    code_hash = source_fingerprint(root)
    _log_stage2("computing per-request resume fingerprints")
    pending: list[tuple[UniAnimateRequest, str]] = []
    resumed = 0
    for index, request in enumerate(requests, start=1):
        _log_stage2(
            f"fingerprint {index}/{len(requests)}: {request.clip_id} "
            f"output={request.output}"
        )
        _log_stage2(
            f"fingerprint {index}/{len(requests)}: hashing 81 control frames "
            f"for {request.clip_id}"
        )
        expected = _stage2_fingerprint(request, model_provenance, code_hash)
        output = Path(request.output)
        status = _read_status(_stage2_status_path(output))
        resumable = (
            status is not None
            and status.get("state") == "complete"
            and status.get("fingerprint") == expected
            and output.is_file()
            and output.stat().st_size > 0
            and status.get("output_sha256") == sha256_file(output)
        )
        if resumable:
            resumed += 1
            _log_stage2(f"resume skip {index}/{len(requests)}: {request.clip_id}")
            continue
        if output.exists() or _stage2_status_path(output).exists():
            if not args.overwrite:
                _log_stage2(
                    f"blocked {index}/{len(requests)}: stale output exists for "
                    f"{request.clip_id} at {output} (pass --overwrite)"
                )
                raise FileExistsError(
                    f"Stale Stage-2 output for {request.clip_id}; pass --overwrite to archive and replace it"
                )
            if not args.dry_run:
                _log_stage2(f"archiving stale output for {request.clip_id}")
                _archive_output(output)
        pending.append((request, expected))
        _log_stage2(f"queued {index}/{len(requests)}: {request.clip_id}")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "requests": len(requests),
                    "pending": len(pending),
                    "resumed": resumed,
                    "write": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if not pending:
        print(json.dumps({"requests": len(requests), "pending": 0, "resumed": resumed}))
        return 0
    print(
        f"[stage2] pending={len(pending)} resumed={resumed} total={len(requests)}",
        flush=True,
    )
    service = UniAnimateBatchService(config)
    errors = 0
    for index, (request, expected) in enumerate(pending, start=1):
        print(
            f"[stage2] start {index}/{len(pending)}: {request.clip_id}",
            flush=True,
        )
        try:
            require_free_disk(
                Path(request.output).parent,
                50.0,
                operation="Stage-2 inference headroom",
            )
            [output] = service.run_batch([request])
            atomic_write_json(
                _stage2_status_path(output),
                {
                    "clip_id": request.clip_id,
                    "state": "complete",
                    "fingerprint": expected,
                    "output": str(output),
                    "output_sha256": sha256_file(output),
                    "model": model_provenance,
                },
            )
            print(
                f"[stage2] done {index}/{len(pending)}: {request.clip_id} -> {output}",
                flush=True,
            )
        except Exception as exc:
            errors += 1
            atomic_write_json(
                _stage2_status_path(Path(request.output)),
                {
                    "clip_id": request.clip_id,
                    "state": "error",
                    "fingerprint": expected,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            print(
                f"[stage2] error {index}/{len(pending)}: {request.clip_id}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
    print(
        json.dumps(
            {
                "requests": len(requests),
                "generated": len(pending) - errors,
                "resumed": resumed,
                "errors": errors,
                "model_loads": service.load_count,
            },
            sort_keys=True,
        )
    )
    return 1 if errors else 0


def _run_subprocess(command: list[str], root: Path, log_path: Path) -> None:
    """Run a cascade stage, streaming child stdout/stderr into ``log_path`` live."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[cascade] starting: {' '.join(command)}", flush=True)
    print(f"[cascade] streaming log -> {log_path}", flush=True)
    with open(log_path, "wb") as log_file:
        process = subprocess.Popen(
            command,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
        )
        assert process.stdout is not None
        while True:
            chunk = process.stdout.read(4096)
            if not chunk:
                break
            log_file.write(chunk)
            log_file.flush()
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        returncode = process.wait()
    if returncode:
        raise RuntimeError(
            f"Cascade subprocess exited {returncode}; see {log_path}"
        )
    print(f"[cascade] finished ok: {log_path.name}", flush=True)


def _manifest_path(value: Any, manifest: Path, root: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidate = (manifest.parent / path).resolve()
    return candidate if candidate.exists() else (root / path).resolve()


def _stage2_rows(
    source_manifest: Path,
    bridge_manifest: Path,
    run_path: Path,
    prompt_source: str,
    seed: int,
    root: Path,
) -> list[dict[str, Any]]:
    source = {str(row["clip_id"]): row for row in _read_rows(source_manifest)}
    rows: list[dict[str, Any]] = []
    for bridge in _read_rows(bridge_manifest):
        clip_id = str(bridge["clip_id"])
        original = source.get(clip_id)
        if original is None:
            raise ValueError(f"{clip_id}: missing from cascade source manifest")
        if not bridge.get("control_dir") or not bridge.get("reference_pose"):
            raise ValueError(f"{clip_id}: bridge did not persist a strict control sequence")
        reference_value = original.get("reference_image")
        if not reference_value:
            raise ValueError(
                f"{clip_id}: cascade manifest must provide reference_image"
            )
        prompt_key = f"{prompt_source}_prompt"
        prompt = str(original.get(prompt_key) or original.get("prompt") or "").strip()
        if not prompt:
            raise ValueError(f"{clip_id}: missing {prompt_key} (and prompt fallback)")
        safe = clip_id.replace("/", "_")
        rows.append(
            {
                "clip_id": clip_id,
                "control_dir": str(bridge["control_dir"]),
                "reference_pose": str(bridge["reference_pose"]),
                "reference_image": str(
                    _manifest_path(reference_value, source_manifest, root)
                ),
                "output": str(run_path / "samples" / safe / "stage2" / "output.mp4"),
                "prompt": prompt,
                "seed": stable_seed(clip_id, base_seed=seed),
                "height": int(original.get("height", 832)),
                "width": int(original.get("width", 480)),
                "num_frames": 81,
                "num_inference_steps": int(original.get("num_inference_steps", 25)),
                "cfg_scale": float(original.get("cfg_scale", 1.5)),
            }
        )
    return rows


def _cascade_adapter_paths(args: Any, root: Path) -> tuple[Path, Path]:
    official = _resolve(root, args.stage2_official_adapter)
    selected_value = getattr(args, "stage2_adapter", None)
    selected = _resolve(root, selected_value) if selected_value else official
    if args.stage2_mode == "frozen" and selected != official:
        raise ValueError(
            "--stage2-mode frozen must use only the official adapter; remove "
            "--stage2-adapter or select --stage2-mode adapted"
        )
    if args.stage2_mode == "adapted":
        if selected_value is None or selected == official:
            raise ValueError(
                "--stage2-mode adapted requires a compact --stage2-adapter "
                "distinct from --stage2-official-adapter"
            )
        from .stage2.adapter_training import validate_training_adapter_checkpoint

        validate_training_adapter_checkpoint(selected)
    return official, selected


def _cascade_commands(args: Any, root: Path, run_path: Path) -> tuple[list[str], list[str], list[str]]:
    official_adapter, selected_adapter = _cascade_adapter_paths(args, root)
    predictions = run_path / "stage1" / "prediction_manifest.jsonl"
    stage1 = [
        sys.executable,
        "-m",
        "dmtbaby",
        "predict-stage1",
        "--root",
        str(root),
        "--teacher-manifest",
        str(_resolve(root, args.manifest)),
        "--checkpoint",
        str(_resolve(root, args.stage1_checkpoint)),
        "--output-root",
        str(run_path / "stage1"),
        "--prompt-source",
        args.prompt_source,
        "--split",
        "all",
        "--seed",
        str(args.seed),
    ]
    bridge = [
        sys.executable,
        "-m",
        "dmtbaby",
        "render-controls",
        "--root",
        str(root),
        "--prediction-manifest",
        str(predictions),
        "--teacher-manifest",
        str(_resolve(root, args.manifest)),
        "--output-root",
        str(run_path / "bridge"),
        "--postprocess",
        args.postprocess,
        "--persist",
        "all",
    ]
    stage2 = [
        sys.executable,
        "-m",
        "dmtbaby",
        "infer-stage2",
        "--root",
        str(root),
        "--manifest",
        str(run_path / "stage2_requests.jsonl"),
        "--official-adapter",
        str(official_adapter),
        "--adapter",
        str(selected_adapter),
    ]
    return stage1, bridge, stage2


def run_cascade(args: Any, root: Path) -> int:
    """Execute Stage 1 and Stage 2 in distinct processes to release GPU memory."""

    source_manifest = _resolve(root, args.manifest)
    stage1_checkpoint = _resolve(root, args.stage1_checkpoint)
    official_adapter, stage2_adapter = _cascade_adapter_paths(args, root)
    for path in (source_manifest, stage1_checkpoint, official_adapter, stage2_adapter):
        if not path.is_file():
            raise FileNotFoundError(path)
    from .stage1.checkpoint import read_checkpoint

    stage1_payload = read_checkpoint(stage1_checkpoint)
    expected_joint_count = 20 if args.joint_scope == "body20" else 62
    actual_joint_count = int(stage1_payload.get("model_config", {}).get("joint_count", 0))
    if actual_joint_count != expected_joint_count:
        raise ValueError(
            f"Stage-1 checkpoint has {actual_joint_count} joints, but cascade "
            f"--joint-scope {args.joint_scope} requires {expected_joint_count}"
        )
    checkpoint_prompt_source = stage1_payload.get("training_config", {}).get(
        "prompt_source"
    )
    if checkpoint_prompt_source != args.prompt_source:
        raise ValueError(
            f"Stage-1 checkpoint prompt source is {checkpoint_prompt_source!r}, "
            f"but cascade requested {args.prompt_source!r}"
        )
    source_rows = _read_rows(source_manifest)
    if args.joint_scope == "body20" and args.postprocess == "constrained":
        # This is supported as the documented implementation milestone; hands
        # remain explicitly invisible at the bridge.
        pass
    if not args.skip_preflight:
        from .preflight import run_preflight

        report = run_preflight(root)
        if not report.ready:
            print(json.dumps(report.as_dict(), sort_keys=True))
            return 2
    config = {
        "format": "dmtbaby.cascade_run.v1",
        "source_manifest": str(source_manifest),
        "prompt_source": args.prompt_source,
        "joint_scope": args.joint_scope,
        "postprocess": args.postprocess,
        "stage1_mode": args.stage1_mode,
        "stage2_mode": args.stage2_mode,
        "seed": int(args.seed),
        "samples": len(source_rows),
        "stage2_official_adapter": str(official_adapter),
        "stage2_adapter": str(stage2_adapter),
    }
    provenance = {
        "code": source_fingerprint(root),
        "inputs": sha256_file(source_manifest),
        "stage1_checkpoint": sha256_file(stage1_checkpoint),
        "stage2_official_adapter": sha256_file(official_adapter),
        "stage2_adapter": sha256_file(stage2_adapter),
        "schema": SCHEMA_ID,
    }
    store = RunStore(_resolve(root, args.runs_root), args.run_id, dry_run=args.dry_run)
    run_path = store.path
    commands = _cascade_commands(args, root, run_path)
    if args.dry_run:
        # Validate the fields needed later without creating the immutable run.
        for row in source_rows:
            if not row.get("reference_image"):
                raise ValueError(f"{row['clip_id']}: missing reference_image")
            prompt = row.get(f"{args.prompt_source}_prompt") or row.get("prompt")
            if not str(prompt or "").strip():
                raise ValueError(f"{row['clip_id']}: missing selected prompt")
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "write": False,
                    "run_path": str(run_path),
                    "commands": commands,
                    "config": config,
                },
                sort_keys=True,
            )
        )
        return 0
    store.open_or_initialize(config, provenance)
    (run_path / "logs").mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(run_path / "source_manifest.jsonl", source_rows)
    stage1_command, bridge_command, stage2_command = commands
    _run_subprocess(stage1_command, root, run_path / "logs/stage1.log")
    _run_subprocess(bridge_command, root, run_path / "logs/bridge.log")
    stage2_rows = _stage2_rows(
        source_manifest,
        run_path / "bridge/control_manifest.jsonl",
        run_path,
        args.prompt_source,
        args.seed,
        root,
    )
    atomic_write_jsonl(run_path / "stage2_requests.jsonl", stage2_rows)
    _run_subprocess(stage2_command, root, run_path / "logs/stage2.log")
    atomic_write_json(
        run_path / "completion.json",
        {
            "state": "complete",
            "samples": len(stage2_rows),
            "config_fingerprint": fingerprint(config),
            "provenance_fingerprint": fingerprint(provenance),
        },
    )
    print(json.dumps({"run": str(run_path), "samples": len(stage2_rows)}))
    return 0
