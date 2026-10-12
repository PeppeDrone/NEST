"""Deterministic hashing, manifests, and atomic filesystem helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object, base_seed: int = 42) -> int:
    payload = canonical_json_bytes([base_seed, *[str(part) for part in parts]])
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFF_FFFF


def fingerprint(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value) + b"\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    atomic_write_bytes(path, payload)


def atomic_write_npz(path: str | Path, arrays: Mapping[str, Any]) -> None:
    """Commit a non-pickled NumPy archive with fsync + atomic rename.

    Object arrays are rejected because they would make the artifact depend on
    pickle at read time.  Callers should keep descriptive metadata in a JSON
    sidecar and use the NPZ strictly for typed tensors.
    """

    import numpy as np

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    normalized: dict[str, Any] = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"NPZ array {name!r} has forbidden object dtype")
        normalized[str(name)] = array
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **normalized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def require_free_disk(
    path: str | Path,
    minimum_gib: float,
    *,
    operation: str,
) -> float:
    """Fail before a write-heavy operation can consume reserved headroom."""

    probe = Path(path).expanduser().resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_gib = shutil.disk_usage(probe).free / 1024**3
    if free_gib < float(minimum_gib):
        raise RuntimeError(
            f"{operation} requires at least {minimum_gib:.1f} GiB free at "
            f"{probe}; found {free_gib:.2f} GiB"
        )
    return free_gib


def environment_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import numpy

        snapshot["numpy"] = numpy.__version__
    except Exception as exc:  # pragma: no cover - diagnostic path
        snapshot["numpy_error"] = repr(exc)
    try:
        import torch

        snapshot.update(
            {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "bf16_supported": bool(torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else False,
            }
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        snapshot["torch_error"] = repr(exc)
    return snapshot
