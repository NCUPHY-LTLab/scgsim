"""Opt-in preparation timing, progress, RSS, and profile output.

Diagnostics are written to process-local JSONL and profile files. They do not
change scientific inputs, outputs, or artifact identities.
"""

from __future__ import annotations

import cProfile
import json
import math
import os
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import TextIO, TypedDict
from uuid import uuid4


class _DiagnosticPayload(TypedDict):
    """JSON-safe worker configuration shared across the native process boundary."""

    output_dir: str
    progress_interval_s: float
    profile: bool
    session_id: str
    parent_span_id: str | None


@dataclass(frozen=True)
class _DiagnosticConfig:
    """Validated settings for one diagnostic session and its child spans."""

    output_dir: str
    progress_interval_s: float
    profile: bool
    session_id: str
    parent_span_id: str | None = None

    def to_payload(self) -> _DiagnosticPayload:
        return {
            "output_dir": self.output_dir,
            "progress_interval_s": self.progress_interval_s,
            "profile": self.profile,
            "session_id": self.session_id,
            "parent_span_id": self.parent_span_id,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> _DiagnosticConfig:
        directory = payload["output_dir"]
        interval = payload["progress_interval_s"]
        profile = payload["profile"]
        session = payload["session_id"]
        parent = payload.get("parent_span_id")
        if not isinstance(directory, str) or not Path(directory).is_absolute():
            raise ValueError("Diagnostic output_dir must be absolute.")
        if isinstance(interval, bool) or not isinstance(interval, (int, float)):
            raise TypeError("Diagnostic progress_interval_s must be numeric.")
        if not math.isfinite(interval) or interval < 0:
            raise ValueError(
                "Diagnostic progress_interval_s must be finite and nonnegative."
            )
        if not isinstance(profile, bool) or not isinstance(session, str) or not session:
            raise TypeError("Invalid diagnostic profile/session configuration.")
        if parent is not None and not isinstance(parent, str):
            raise TypeError("Invalid diagnostic parent span.")
        return cls(directory, float(interval), profile, session, parent)


@dataclass
class _Span:
    """One timed operation and its parent in the current diagnostic tree."""

    span_id: str
    parent_span_id: str | None
    name: str
    started_wall: float = field(default_factory=time.perf_counter)
    started_cpu: float = field(default_factory=time.process_time)
    last_progress: float = field(default_factory=time.monotonic)


@dataclass
class _Recorder:
    """Serialize concurrent stage events into one process-local JSONL stream."""

    config: _DiagnosticConfig
    stream: TextIO
    errors: list[str] = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)

    def emit(self, event: str, span: _Span, **values: object) -> None:
        with self.lock:
            try:
                row = {
                    "event": event,
                    "session_id": self.config.session_id,
                    "span_id": span.span_id,
                    "parent_span_id": span.parent_span_id,
                    "stage": span.name,
                    "pid": os.getpid(),
                    "wall_time_ns": time.time_ns(),
                    "elapsed_wall_s": time.perf_counter() - span.started_wall,
                    "elapsed_process_cpu_s": time.process_time() - span.started_cpu,
                    **_rss(),
                    **values,
                }
                self.stream.write(json.dumps(row, allow_nan=False) + "\n")
                self.stream.flush()
            except Exception as error:  # noqa: BLE001 -- diagnostics must never replace preparation outcomes
                self.errors.append(type(error).__name__)


def _rss() -> dict[str, object]:
    try:
        import resource  # noqa: PLC0415 -- unavailable on some supported platforms

        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            value = raw
        elif sys.platform.startswith("linux"):
            value = raw * 1024
        else:
            return {
                "process_peak_rss_bytes": None,
                "rss_status": "unsupported_platform",
            }
        return {"process_peak_rss_bytes": value, "rss_status": "available"}
    except (ImportError, OSError, AttributeError):
        return {"process_peak_rss_bytes": None, "rss_status": "unavailable"}


_recorder: ContextVar[_Recorder | None] = ContextVar("scgsim_diagnostics", default=None)
_span: ContextVar[_Span | None] = ContextVar("scgsim_diagnostic_span", default=None)


@contextmanager
def diagnostic_stage(
    name: str, *, counts: dict[str, int] | None = None
) -> Iterator[None]:
    """Record the start, progress context and outcome of one diagnostic stage."""
    recorder = _recorder.get()
    if recorder is None:
        yield
        return
    parent = _span.get()
    span = _Span(
        uuid4().hex, parent.span_id if parent else recorder.config.parent_span_id, name
    )
    token = _span.set(span)
    recorder.emit("start", span, counts=counts)
    try:
        yield
    except BaseException as error:
        recorder.emit("error", span, error_type=type(error).__name__)
        raise
    else:
        recorder.emit("end", span)
    finally:
        _span.reset(token)


def progress(completed: int, total: int | None = None) -> None:
    """Emit throttled progress for the current diagnostic stage."""
    recorder, span = _recorder.get(), _span.get()
    if recorder is None or span is None:
        return
    now = time.monotonic()
    if now - span.last_progress >= recorder.config.progress_interval_s:
        span.last_progress = now
        recorder.emit("progress", span, completed=completed, total=total)


def worker_config() -> _DiagnosticPayload | None:
    """Capture diagnostic settings and the parent span for a worker."""
    recorder = _recorder.get()
    if recorder is None:
        return None
    payload = recorder.config.to_payload()
    span = _span.get()
    payload["parent_span_id"] = span.span_id if span else recorder.config.parent_span_id
    return payload


@contextmanager
def _diagnostic_session(config: _DiagnosticConfig) -> Iterator[None]:
    directory = Path(config.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if config.profile and sys.getprofile() is not None:
        raise RuntimeError("A current-thread profiler is already active.")
    stem = f"{config.session_id}-{os.getpid()}-{uuid4().hex}"
    # Exclusive creation rejects invalid/unwritable destinations before preparation.
    stream = (directory / f"{stem}.jsonl").open("x", encoding="utf-8")
    recorder = _Recorder(config, stream)
    token = _recorder.set(recorder)
    span_token = _span.set(None)
    profiler = cProfile.Profile() if config.profile else None
    primary: BaseException | None = None
    try:
        if profiler is not None:
            profiler.enable()
        with diagnostic_stage("preparation.session"):
            yield
    except BaseException as error:
        primary = error
        raise
    finally:
        if profiler is not None:
            profiler.disable()
            try:
                # Reserve a unique path without replacing an earlier profile.
                path = directory / f"{stem}.pstats"
                with path.open("xb"):
                    pass
                profiler.dump_stats(str(path))
            except Exception as error:  # noqa: BLE001 -- retain the user's error and report cleanup failure as a note
                recorder.errors.append(type(error).__name__)
        _span.reset(span_token)
        _recorder.reset(token)
        try:
            stream.close()
        except Exception as error:  # noqa: BLE001 -- retain the user's error and report cleanup failure as a note
            recorder.errors.append(type(error).__name__)
        if recorder.errors:
            detail = "Preparation diagnostics failed: " + ", ".join(recorder.errors)
            if primary is not None:
                primary.add_note(detail)
            else:
                raise RuntimeError(detail)


@contextmanager
def preparation_diagnostics(
    output_dir: str | os.PathLike[str],
    *,
    progress_interval_s: float = 1.0,
    profile: bool = False,
) -> Iterator[None]:
    """Write per-process JSONL diagnostics for calls inside this scope.

    Profiling is opt-in and applies only to the calling thread. RSS is the
    process lifetime peak, not aggregate memory of a job or a stage allocation.
    Scientific snapshots, mesh controls and artifact identities are unchanged.
    """
    config = _DiagnosticConfig.from_payload(
        {
            "output_dir": str(Path(output_dir).resolve()),
            "progress_interval_s": progress_interval_s,
            "profile": profile,
            "session_id": uuid4().hex,
        }
    )
    with _diagnostic_session(config):
        yield


@contextmanager
def worker_diagnostics(payload: Mapping[str, object] | None) -> Iterator[None]:
    """Run a worker within the supplied diagnostic session."""
    if payload is None:
        yield
        return
    with _diagnostic_session(_DiagnosticConfig.from_payload(payload)):
        yield
