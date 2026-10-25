"""Immutable v1 batch records and pure file identity checks; standard library only."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Literal, TypeAlias, cast

JsonValue: TypeAlias = None | bool | int | float | str | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]
JsonObject: TypeAlias = Mapping[str, JsonValue]
Number: TypeAlias = int | float
AttemptState: TypeAlias = Literal["not_started", "starting", "running", "external_process_terminal", "unsettled", "failed"]
PACKAGE_SCHEMA = "scgsim.aedt.native-batch-package.v1"
ATTEMPT_SCHEMA = "scgsim.aedt.native-batch-attempt.v1"
MANIFEST = "native-batch-package.json"


def _freeze(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in cast(Sequence[object], value))
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in cast(Mapping[object, object], value).items():
            if not isinstance(key, str):
                raise TypeError("batch JSON object keys must be strings")
            result[key] = _freeze(item)
        return MappingProxyType(result)
    raise TypeError("batch record contains a non-JSON value")


def _thaw(value: JsonValue) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def object_record(value: object) -> JsonObject:
    frozen = _freeze(value)
    if not isinstance(frozen, Mapping):
        raise TypeError("batch record requires a JSON object")
    return frozen


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("batch record requires a string")
    return value


def _number(value: object) -> Number:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("batch record requires a numeric timestamp")
    return value


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("batch record requires an integer")
    return value


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("batch record requires a string array")
    return tuple(_text(item) for item in cast(Sequence[object], value))


def _extras(value: JsonObject, known: set[str]) -> JsonObject:
    return MappingProxyType({key: item for key, item in value.items() if key not in known})


def object_payload(value: JsonObject) -> dict[str, object]:
    return cast(dict[str, object], _thaw(value))


def read_json(path: Path) -> JsonObject:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    return object_record(value)


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(object_payload(object_record(value)), stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inside(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError("batch member must be a contained relative path")
    path = root / rel
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("batch member escapes its root or is a symlink")
    return path


def fresh(path: str | Path, *sources: Path) -> Path:
    result = Path(path).expanduser().resolve()
    for source in sources:
        if result.is_relative_to(source) or source.is_relative_to(result):
            raise ValueError("batch output and input roots must not overlap")
    result.mkdir(parents=True, exist_ok=False)
    return result


@dataclass(frozen=True)
class Member:
    path: str
    bytes: int
    sha256: str

    @classmethod
    def from_payload(cls, value: object) -> Member:
        record = object_record(value)
        if set(record) != {"path", "bytes", "sha256"}:
            raise ValueError("batch inventory member is not canonical")
        return cls(_text(record["path"]), _integer(record["bytes"]), _text(record["sha256"]))

    def to_payload(self) -> dict[str, object]:
        return {"path": self.path, "bytes": self.bytes, "sha256": self.sha256}


def members(value: object) -> tuple[Member, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("batch inventory must be an array")
    return tuple(Member.from_payload(item) for item in cast(Sequence[object], value))


def inventory(root: Path) -> tuple[Member, ...]:
    records: list[Member] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("batch inventory contains a symlink or escaping member")
        if path.is_file():
            records.append(Member(path.relative_to(root).as_posix(), path.stat().st_size, sha256(path)))
    return tuple(records)


def verify_members(root: Path, records: Sequence[Member]) -> None:
    names: set[str] = set()
    for item in records:
        path = inside(root, item.path)
        if item.path in names:
            raise ValueError("batch inventory repeats a member")
        names.add(item.path)
        if not path.is_file() or path.stat().st_size != item.bytes or sha256(path) != item.sha256:
            raise RuntimeError(f"batch member identity differs: {item.path}")


def copy_members(source: Path, target: Path, records: Sequence[Member]) -> None:
    verify_members(source, records)
    for item in records:
        destination = inside(target, item.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(inside(source, item.path), destination)
    verify_members(target, records)


@dataclass(frozen=True)
class Resources:
    cores: int
    ram_limit_percent: int

    @classmethod
    def from_payload(cls, value: object) -> Resources:
        record = object_record(value)
        if set(record) != {"cores", "ram_limit_percent"}:
            raise ValueError("batch resources require cores and ram_limit_percent")
        try:
            cores, ram = _integer(record["cores"]), _integer(record["ram_limit_percent"])
        except TypeError as exc:
            raise ValueError("batch resource limits must be integers") from exc
        if cores < 1:
            raise ValueError("batch cores must be a positive integer")
        if not 1 <= ram <= 99:
            raise ValueError("native MachineList RAM percentage must be in 1..99")
        return cls(cores, ram)

    def to_payload(self) -> dict[str, object]:
        return {"cores": self.cores, "ram_limit_percent": self.ram_limit_percent}


@dataclass(frozen=True)
class Recipe:
    executable: str
    mode: str
    allocation: str
    target: str
    extra: JsonObject = field(default_factory=lambda: MappingProxyType({}), repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", object_record(self.extra))

    @classmethod
    def from_payload(cls, value: object) -> Recipe:
        record = object_record(value)
        return cls(executable=_text(record["executable"]), mode=_text(record["mode"]),
                   allocation=_text(record["allocation"]), target=_text(record["target"]),
                   extra=_extras(record, {"executable", "mode", "allocation", "target"}))

    def to_payload(self) -> dict[str, object]:
        return {**object_payload(self.extra), "executable": self.executable, "mode": self.mode,
                "allocation": self.allocation, "target": self.target}


@dataclass(frozen=True)
class PackageManifest:
    package_id: str
    created_at_unix: Number
    source_location: str
    requested_aedt_version: str
    prepared_connected: JsonObject
    spec_sha256: str
    model_sha256: str
    source_sha256: str
    project: str
    results: str
    design: str
    setup: str
    resources: Resources
    recipe: Recipe
    declared_dependencies: tuple[str, ...]
    members: tuple[Member, ...]
    extra: JsonObject = field(default_factory=lambda: MappingProxyType({}), repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "prepared_connected", object_record(self.prepared_connected))
        object.__setattr__(self, "extra", object_record(self.extra))
        object.__setattr__(self, "members", tuple(self.members))
        object.__setattr__(self, "declared_dependencies", tuple(self.declared_dependencies))

    @classmethod
    def from_payload(cls, value: object) -> PackageManifest:
        record = object_record(value)
        if record.get("schema_version") != PACKAGE_SCHEMA or record.get("status") != "prepared_native":
            raise ValueError("native batch package is not prepared")
        known = {"schema_version", "status", *cls.__dataclass_fields__} - {"extra"}
        return cls(
            package_id=_text(record["package_id"]), created_at_unix=_number(record["created_at_unix"]),
            source_location=_text(record["source_location"]),
            requested_aedt_version=_text(record["requested_aedt_version"]),
            prepared_connected=object_record(record["prepared_connected"]),
            spec_sha256=_text(record["spec_sha256"]), model_sha256=_text(record["model_sha256"]),
            source_sha256=_text(record["source_sha256"]), project=_text(record["project"]),
            results=_text(record["results"]), design=_text(record["design"]), setup=_text(record["setup"]),
            resources=Resources.from_payload(record["resources"]), recipe=Recipe.from_payload(record["recipe"]),
            declared_dependencies=_strings(record["declared_dependencies"]), members=members(record["members"]),
            extra=_extras(record, known),
        )

    def to_payload(self) -> dict[str, object]:
        return {**object_payload(self.extra), "schema_version": PACKAGE_SCHEMA, "status": "prepared_native",
                "package_id": self.package_id, "created_at_unix": self.created_at_unix,
                "source_location": self.source_location, "requested_aedt_version": self.requested_aedt_version,
                "prepared_connected": object_payload(self.prepared_connected), "spec_sha256": self.spec_sha256,
                "model_sha256": self.model_sha256, "source_sha256": self.source_sha256,
                "project": self.project, "results": self.results, "design": self.design, "setup": self.setup,
                "resources": self.resources.to_payload(), "recipe": self.recipe.to_payload(),
                "declared_dependencies": list(self.declared_dependencies),
                "members": [item.to_payload() for item in self.members]}


@dataclass(frozen=True)
class BatchAttempt:
    attempt_id: str
    package_sha256: str
    package_id: str
    status: AttemptState
    requested_aedt_version: str
    actual_aedt_version: str | None
    version_observation: str
    execution_location: JsonObject
    resources_requested: Resources
    resource_semantics: str
    started_at_unix: Number
    argv: tuple[str, ...] | None = None
    executable_sha256: str | None = None
    child_pid: int | None = None
    returncode: int | None = None
    finished_at_unix: Number | None = None
    work_members: tuple[Member, ...] | None = None
    raw_evidence: tuple[Member, ...] | None = None
    error: str | None = None
    extra: JsonObject = field(default_factory=lambda: MappingProxyType({}), repr=False)
    present: frozenset[str] = field(default_factory=lambda: frozenset[str](), repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "execution_location", object_record(self.execution_location))
        object.__setattr__(self, "extra", object_record(self.extra))
        object.__setattr__(self, "present", frozenset(self.present))
        if self.argv is not None:
            object.__setattr__(self, "argv", tuple(self.argv))
        if self.work_members is not None:
            object.__setattr__(self, "work_members", tuple(self.work_members))
        if self.raw_evidence is not None:
            object.__setattr__(self, "raw_evidence", tuple(self.raw_evidence))

    @classmethod
    def from_payload(cls, value: object) -> BatchAttempt:
        record = object_record(value)
        if record.get("schema_version") != ATTEMPT_SCHEMA:
            raise ValueError("batch attempt schema is unsupported")
        state = _text(record["status"])
        if state not in {"not_started", "starting", "running", "external_process_terminal", "unsettled", "failed"}:
            raise ValueError("batch attempt state is unsupported")
        known = {"schema_version", *cls.__dataclass_fields__} - {"extra", "present"}
        def optional_text(key: str) -> str | None:
            return None if record.get(key) is None else _text(record[key])
        def optional_int(key: str) -> int | None:
            return None if record.get(key) is None else _integer(record[key])
        return cls(
            attempt_id=_text(record["attempt_id"]), package_sha256=_text(record["package_sha256"]),
            package_id=_text(record["package_id"]), status=cast(AttemptState, state),
            requested_aedt_version=_text(record["requested_aedt_version"]),
            actual_aedt_version=optional_text("actual_aedt_version"),
            version_observation=_text(record["version_observation"]),
            execution_location=object_record(record["execution_location"]),
            resources_requested=Resources.from_payload(record["resources_requested"]),
            resource_semantics=_text(record["resource_semantics"]), started_at_unix=_number(record["started_at_unix"]),
            argv=None if record.get("argv") is None else _strings(record["argv"]),
            executable_sha256=optional_text("executable_sha256"), child_pid=optional_int("child_pid"),
            returncode=optional_int("returncode"),
            finished_at_unix=None if record.get("finished_at_unix") is None else _number(record["finished_at_unix"]),
            work_members=None if record.get("work_members") is None else members(record["work_members"]),
            raw_evidence=None if record.get("raw_evidence") is None else members(record["raw_evidence"]),
            error=optional_text("error"), extra=_extras(record, known), present=frozenset(record),
        )

    def to_payload(self) -> dict[str, object]:
        value = {**object_payload(self.extra), "schema_version": ATTEMPT_SCHEMA, "attempt_id": self.attempt_id,
                 "package_sha256": self.package_sha256, "package_id": self.package_id, "status": self.status,
                 "requested_aedt_version": self.requested_aedt_version, "actual_aedt_version": self.actual_aedt_version,
                 "version_observation": self.version_observation, "execution_location": object_payload(self.execution_location),
                 "resources_requested": self.resources_requested.to_payload(), "resource_semantics": self.resource_semantics,
                 "started_at_unix": self.started_at_unix}
        optional: dict[str, object] = {
            "argv": None if self.argv is None else list(self.argv), "executable_sha256": self.executable_sha256,
            "child_pid": self.child_pid, "returncode": self.returncode, "finished_at_unix": self.finished_at_unix,
            "work_members": None if self.work_members is None else [item.to_payload() for item in self.work_members],
            "raw_evidence": None if self.raw_evidence is None else [item.to_payload() for item in self.raw_evidence],
            "error": self.error,
        }
        value.update({key: item for key, item in optional.items() if item is not None or key in self.present})
        return value


@dataclass(frozen=True)
class CollectionContext:
    package_sha256: str
    attempt_sha256: str

    @classmethod
    def from_payload(cls, value: object) -> CollectionContext:
        record = object_record(value)
        if set(record) != {"package_sha256", "attempt_sha256"}:
            raise ValueError("batch collection context fields differ")
        return cls(_text(record["package_sha256"]), _text(record["attempt_sha256"]))

    def to_payload(self) -> dict[str, object]:
        return {"package_sha256": self.package_sha256, "attempt_sha256": self.attempt_sha256}


def read_package(root: Path) -> PackageManifest:
    value = PackageManifest.from_payload(read_json(root / MANIFEST))
    verify_members(root, value.members)
    observed = tuple(item for item in inventory(root) if item.path not in {MANIFEST, "native_batch.tar.gz"})
    if observed != value.members:
        raise RuntimeError("batch package contains undeclared or changed files")
    recipe = value.recipe
    if recipe.mode != "prepared_project_batchsolve_ppe" or recipe.allocation != "one_host_native_auto":
        raise ValueError("unsupported native batch execution recipe")
    if recipe.target != f"{value.design}:Nominal:{value.setup}":
        raise ValueError("batch target differs from prepared design/setup")
    if sha256(root / "inputs/aedt_spec.json") != value.spec_sha256:
        raise RuntimeError("batch spec identity differs")
    spec = read_json(root / "inputs/aedt_spec.json")
    prepared = read_json(root / "native-preparation-receipt.json")
    project = object_record(spec["project"])
    run = object_record(spec["run_control"])
    expected_project = f"{_text(project['name'])}.aedt"
    if (value.design != project["design"] or value.setup != run["setup_name"]
            or value.project != expected_project
            or value.requested_aedt_version != object_record(spec["aedt"])["requested_version"]):
        raise RuntimeError("batch execution identity differs from packaged request")
    if (prepared.get("status") != "native_preparation_only" or prepared.get("release") != {"ok": True}
            or object_record(prepared.get("save")).get("ok") is not True
            or prepared.get("solver_invoked") is not False
            or object_record(prepared.get("source")).get("spec_sha256") != value.spec_sha256
            or prepared.get("project") != expected_project
            or object_record(prepared.get("setup")).get("name") != value.setup
            or object_record(prepared.get("requested")).get("aedt_version") != value.requested_aedt_version
            or prepared.get("connected") != value.prepared_connected
            or sha256(inside(root / "native", expected_project)) != object_record(prepared["save"])["project_sha256"]):
        raise RuntimeError("batch execution identity differs from released native preparation")
    return value


def read_attempt(package_root: Path, attempt_root: Path, package: PackageManifest) -> BatchAttempt:
    value = BatchAttempt.from_payload(read_json(attempt_root / "batch-attempt.json"))
    if (value.status != "external_process_terminal" or value.returncode != 0
            or value.package_sha256 != sha256(package_root / MANIFEST) or value.package_id != package.package_id):
        raise RuntimeError("batch attempt is not a matching settled zero-exit process")
    if value.work_members is None or value.raw_evidence is None:
        raise RuntimeError("batch terminal attempt lacks returned inventories")
    verify_members(attempt_root / "work", value.work_members)
    if inventory(attempt_root / "work") != value.work_members:
        raise RuntimeError("returned batch inventory changed after process settlement")
    verify_members(attempt_root, value.raw_evidence)
    if (not inside(attempt_root / "work", package.project).is_file()
            or not inside(attempt_root / "work", package.results).is_dir()):
        raise RuntimeError("batch return lacks the saved project or complete results directory")
    return value


def collection_source(
    run_dir: Path, context: CollectionContext,
) -> tuple[PackageManifest, BatchAttempt, JsonObject]:
    package_root = run_dir / "batch-source/package"
    attempt_root = run_dir / "batch-source/attempt"
    if (context != CollectionContext.from_payload(read_json(run_dir / "metadata/batch-collection.json"))
            or context.package_sha256 != sha256(package_root / MANIFEST)
            or context.attempt_sha256 != sha256(attempt_root / "batch-attempt.json")):
        raise RuntimeError("native batch collection parent identity differs")
    package = read_package(package_root)
    attempt = read_attempt(package_root, attempt_root, package)
    prepared = read_json(package_root / "native-preparation-receipt.json")
    source = object_record(prepared.get("source"))
    if (prepared.get("status") != "native_preparation_only" or prepared.get("release") != {"ok": True}
            or object_record(prepared.get("save")).get("ok") is not True or prepared.get("solver_invoked") is not False
            or source.get("spec_sha256") != package.spec_sha256
            or source.get("planar_model_sha256") != package.model_sha256
            or source.get("planar_source_sha256") != package.source_sha256
            or sha256(package_root / "native" / package.project) != object_record(prepared["save"])["project_sha256"]):
        raise RuntimeError("batch parent is not the released original prepared native model")
    for item in package.members:
        if item.path.startswith("inputs/") and item.path != "inputs/metadata/aedt_run_receipt.json":
            local = inside(run_dir, item.path.removeprefix("inputs/"))
            if not local.is_file() or sha256(local) != item.sha256:
                raise RuntimeError("collection input differs from original prepared source")
    return package, attempt, prepared
