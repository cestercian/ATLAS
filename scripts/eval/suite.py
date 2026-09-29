"""A held-out suite on disk, checked against the manifest it was frozen with."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

MODES = ("work", "question")
RUNTIMES = ("python3.13", "go", "node")


class SuiteError(Exception):
    """The suite on disk is malformed, or is not the one that was frozen."""


@dataclass(frozen=True)
class Task:
    id: str
    dir: Path
    mode: str
    runtime: str
    network: bool
    grader_timeout_s: int

    @property
    def prompt(self) -> str:
        return (self.dir / "prompt.md").read_text()

    @property
    def seed(self) -> Path:
        return self.dir / "seed"

    @property
    def grader(self) -> Path:
        return self.dir / "grade"

    def control(self, kind: str) -> Path:
        return self.dir / "controls" / kind


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_suite(root: Path) -> list[Task]:
    """Every task in the suite, after each file is checked against suite.json.

    The root is made absolute here: the grader mounts paths under it with
    `docker run -v`, and docker reads a relative host path as a volume name."""
    root = Path(root).resolve()
    manifest = _read_json(root / "suite.json")
    tasks = [_load_task(root / "tasks" / entry["id"], entry)
             for entry in manifest.get("tasks", [])]
    if not tasks:
        raise SuiteError(f"{root / 'suite.json'} lists no tasks")
    return tasks


def _load_task(task_dir: Path, entry: dict) -> Task:
    _check_frozen(task_dir, entry.get("files") or {})
    meta = _read_json(task_dir / "task.json")
    task = Task(id=str(meta.get("id", "")), dir=task_dir,
                mode=meta.get("mode", "work"), runtime=meta.get("runtime", ""),
                network=meta.get("network", False),
                grader_timeout_s=meta.get("grader_timeout_s", 120))
    _check_task(task, entry["id"])
    return task


def _check_frozen(task_dir: Path, frozen: dict) -> None:
    """Every frozen file is present and unchanged, and nothing was added."""
    on_disk = {p.relative_to(task_dir).as_posix()
               for p in task_dir.rglob("*") if p.is_file()}
    added = sorted(on_disk - set(frozen))
    missing = sorted(set(frozen) - on_disk)
    changed = sorted(f for f in set(frozen) & on_disk
                     if sha256_file(task_dir / f) != frozen[f])
    problems = ([f"added after the freeze: {f}" for f in added]
                + [f"missing: {f}" for f in missing]
                + [f"changed since the freeze: {f}" for f in changed])
    if problems:
        raise SuiteError(f"task {task_dir.name}: " + "; ".join(problems))


def _check_task(task: Task, listed_id: str) -> None:
    problems = []
    if task.id != listed_id:
        problems.append(f"task.json id {task.id!r} is not the listed {listed_id!r}")
    if task.mode not in MODES:
        problems.append(f"mode {task.mode!r} is not one of {MODES}")
    if task.runtime not in RUNTIMES:
        problems.append(f"runtime {task.runtime!r} is not one of {RUNTIMES}")
    if not isinstance(task.network, bool):
        problems.append("network must be true or false")
    if not isinstance(task.grader_timeout_s, int) or task.grader_timeout_s <= 0:
        problems.append("grader_timeout_s must be a positive whole number")
    for need in ("prompt.md", "grade"):
        if not (task.dir / need).is_file():
            problems.append(f"no {need}")
    if problems:
        raise SuiteError(f"task {listed_id}: " + "; ".join(problems))


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise SuiteError(f"cannot read {path}: {e}") from e
