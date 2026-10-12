"""Immutable run manifests and fingerprint-aware resume decisions."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .io import (
    atomic_write_json,
    environment_snapshot,
    fingerprint,
    read_json,
    sha256_file,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ArtifactFingerprint:
    config: str
    code: str
    inputs: str
    checkpoint: str
    schema: str
    renderer: str

    @property
    def combined(self) -> str:
        return fingerprint(asdict(self))


@dataclass
class SampleStatus:
    clip_id: str
    state: str
    fingerprint: str
    updated_at: str = field(default_factory=utc_now)
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class RunStore:
    """Owns run-scoped artifacts without treating mere existence as success."""

    def __init__(self, root: str | Path, run_id: str, *, dry_run: bool = False):
        if not run_id or run_id in {".", ".."} or "/" in run_id:
            raise ValueError(f"Unsafe run_id: {run_id!r}")
        self.root = Path(root).resolve()
        self.run_id = run_id
        self.path = self.root / run_id
        self.dry_run = dry_run

    def initialize(self, config: Mapping[str, Any], provenance: Mapping[str, Any]) -> None:
        if self.dry_run:
            return
        self.path.mkdir(parents=True, exist_ok=False)
        atomic_write_json(self.path / "resolved_config.json", dict(config))
        atomic_write_json(
            self.path / "run_manifest.json",
            {
                "run_id": self.run_id,
                "created_at": utc_now(),
                "config_hash": fingerprint(config),
                "provenance_hash": fingerprint(provenance),
                "provenance": dict(provenance),
                "environment": environment_snapshot(),
            },
        )

    def open_or_initialize(
        self, config: Mapping[str, Any], provenance: Mapping[str, Any]
    ) -> bool:
        """Create an immutable run or verify an exact-match resume.

        Returns ``True`` for an existing compatible run and ``False`` for a
        newly initialized run.  Dry-run validates an existing run when one is
        present but never creates anything.
        """

        if not self.path.exists():
            self.initialize(config, provenance)
            return False
        if not self.path.is_dir():
            raise ValueError(f"Run path is not a directory: {self.path}")
        config_path = self.path / "resolved_config.json"
        manifest_path = self.path / "run_manifest.json"
        if not config_path.is_file() or not manifest_path.is_file():
            raise ValueError(f"Existing run is missing immutable metadata: {self.path}")
        stored_config = read_json(config_path)
        manifest = read_json(manifest_path)
        expected_config_hash = fingerprint(config)
        expected_provenance_hash = fingerprint(provenance)
        if stored_config != dict(config) or manifest.get("config_hash") != expected_config_hash:
            raise ValueError(
                f"Run {self.run_id!r} was created with a different resolved configuration"
            )
        if (
            manifest.get("provenance") != dict(provenance)
            or manifest.get("provenance_hash") != expected_provenance_hash
        ):
            raise ValueError(
                f"Run {self.run_id!r} was created with different source/checkpoint provenance"
            )
        return True

    def sample_dir(self, clip_id: str) -> Path:
        safe = clip_id.replace("/", "_").replace(os.sep, "_")
        if safe in {"", ".", ".."}:
            raise ValueError(f"Unsafe clip_id: {clip_id!r}")
        return self.path / "samples" / safe

    def read_status(self, clip_id: str) -> SampleStatus | None:
        path = self.sample_dir(clip_id) / "status.json"
        if not path.is_file():
            return None
        payload = read_json(path)
        return SampleStatus(**payload)

    def can_resume(self, clip_id: str, expected_fingerprint: str) -> bool:
        status = self.read_status(clip_id)
        if status is None or status.state != "complete" or status.fingerprint != expected_fingerprint:
            return False
        return all((self.sample_dir(clip_id) / relative).is_file() for relative in status.artifacts.values())

    def write_status(self, status: SampleStatus) -> None:
        if self.dry_run:
            return
        directory = self.sample_dir(status.clip_id)
        directory.mkdir(parents=True, exist_ok=True)
        status.updated_at = utc_now()
        atomic_write_json(directory / "status.json", status.as_dict())


def source_fingerprint(root: str | Path) -> str:
    """Hash executable source and locked configuration, excluding prose/snapshots.

    Editing a README must not invalidate 2,593 teacher artifacts.  Conversely,
    changes to executable Python, resolved experiment configuration, the
    patient split, external-source provenance, or the reviewed UniAnimate patch
    must invalidate resume decisions.
    """
    base = Path(root).resolve()
    entries: list[tuple[str, str]] = []
    source_roots = {
        "dmtbaby": {".py"},
        "CodeStage1": {".py", ".json", ".yaml", ".yml"},
        "CodeStage2": {".py", ".json", ".yaml", ".yml"},
        "configs": {".toml", ".json", ".yaml", ".yml"},
    }
    for relative_root, suffixes in source_roots.items():
        directory = base / relative_root
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in suffixes:
                entries.append((str(path.relative_to(base)), sha256_file(path)))
    for relative in (
        "pyproject.toml",
        "requirements.txt",
        "reproducibility/environment-cascade.yml",
        "reproducibility/patient_split_v1.json",
        "reproducibility/source-provenance.json",
        "reproducibility/patches/unianimate-local.patch",
    ):
        path = base / relative
        if path.is_file():
            entries.append((relative, sha256_file(path)))
    return fingerprint(entries)
