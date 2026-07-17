#!/usr/bin/env python3
"""Deterministic multi-parameter autotuner for the DS4 Metal runtime.

The tuner performs coordinate ascent over an explicit allow-list of runtime
knobs.  Every candidate is executed in a separate DS4Demo process, checked
against bounded full-vocabulary logit traces, and promoted only after an
order-balanced ABBA confirmation.  Lossy model changes are intentionally not
part of the search space.

The script uses only the Python standard library and the comparison helpers in
``metal_ab_compare.py``.  Run ``--self-test`` without a model for fast tests.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from metal_ab_compare import compare_traces, parse_performance, write_synthetic


SCHEMA = 2
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BINARY = ROOT / ".build" / "release" / "DS4Demo"
TRANSIENT_DS4_KEYS = {
    "DS4_AB_TRACE",
    "DS4_AB_TRACE_FRAMES",
    "DS4_DIAG",
    "DS4_USAGE_FILE",
    "DS4_WARMUP",
    "DS4_DEMO_TEMPERATURE",
    "DS4_DEMO_TOP_K",
    "DS4_DEMO_TOP_P",
    "DS4_DEMO_MIN_P",
    "DS4_DEMO_REPEAT_PENALTY",
    "DS4_SPEC_K",
    "DS4_SPEC_DRAFT",
    "DS4_SPEC_DRAFT_EXPERTS",
}


M1PRO16_MIXED_PRESET: dict[str, str] = {
    "DS4_DEMO_CONTEXT": "4096",
    "DS4_EXPERT_CACHE_SLOTS": "22",
    "DS4_EXPERT_CACHE_UNIFORM": "1",
    "DS4_MULTI_QUANT_CACHE": "1",
    "DS4_EXPERT_PREAD": "1",
    "DS4_PREAD_SPLIT": "1",
    "DS4_WILLNEED_EXPERTS": "1",
    "DS4_EXPERT_BUNDLE": "0",
    "DS4_MTLIO": "0",
    "DS4_POOL_INTERLEAVE": "1",
    "DS4_PREFETCH": "0",
    "DS4_PREFETCH_EXPERTS": "0",
    "DS4_EXPERT_LOOKAHEAD": "0",
    "DS4_ASYNC_FFN": "1",
    "DS4_ASYNC_ROUTE": "1",
    "DS4_PREFILL_UNION": "256",
    "DS4_PREFILL_CHUNK": "512",
    "DS4_PREFILL_ROUTE_BATCH": "32",
    "DS4_PREFILL_FFN_BATCH": "1",
    "DS4_PREFILL_MM": "0",
    "DS4_ACTIVE_EXPERTS": "6",
    # This machine needs constant-size raw KV.  The autotuner never changes it.
    "DS4_RAW_RING": "1",
    "DS4_RESIDENT_DENSE": "0",
    "DS4_DENSE_STREAM": "1",
    "DS4_DENSE_AHEAD": "2",
    "DS4_DENSE_Q4": "1",
    "DS4_QKV_Q4": "1",
    "DS4_SHARED_Q4": "1",
    "DS4_RESIDENT_COMP": "1",
    "DS4_COMP_Q8": "0",
    "DS4_LAZY_IDX": "1",
    "DS4_GPU_INDEXER_TOPK": "1",
    "DS4_ADAPTIVE_SPLITK": "1",
    "DS4_DENSE_Q4_KERNEL": "1",
    "DS4_FUSED_HC": "1",
    "DS4_FUSED_ROUTER_PROBS": "1",
    "DS4_FUSED_ROUTER_FINALIZE": "1",
    "DS4_FUSED_COMP_PROJ": "1",
    "DS4_VECTOR_COPY": "0",
    "DS4_FLASH_KV_STAGE": "0",
    "DS4_ROPE_PAIR": "0",
    # No effect while ROPE_PAIR=0; when the candidate is enabled, exercise the
    # intended GPU-affine pair path rather than the host-position fallback.
    "DS4_ROPE_AFFINE": "1",
    "DS4_Q8_NSG": "4",
    "DS4_MOE_NSG": "4",
    "DS4_DENSE_Q4_NSG": "4",
    "DS4_MLOCK": "1",
    "DS4_PROFILE_ROUTE": "0",
    "DS4_SPEC_VERIFY_BATCH": "0",
    "DS4_DEMO_REPEAT_LAST_N": "64",
}


@dataclass(frozen=True)
class Parameter:
    name: str
    values: tuple[str, ...]
    default: str
    metric: str = "decodeSteadyTps"
    quality: str = "exact"
    profiles: tuple[str, ...] = ("standard", "full")
    description: str = ""
    requires: tuple[tuple[str, str], ...] = ()
    requires_nonzero: tuple[str, ...] = ()
    usage_dependent: bool = False
    min_prompt_tokens: int = 0
    min_prompt_equals_value: bool = False
    memory_risk: bool = False
    # ``sweep`` is for small, non-monotonic hardware grids (queue depth/NSG).
    # ``walk`` is for genuinely ordered resource knobs.
    search: str = "walk"


def string_values(values: Iterable[int]) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


PARAMETERS: tuple[Parameter, ...] = (
    Parameter(
        "DS4_PREAD_SPLIT", string_values(range(1, 9)), "1",
        profiles=("io", "standard", "full"),
        description="NVMe queue depth for direct expert pread",
        requires=(("DS4_EXPERT_PREAD", "1"), ("DS4_EXPERT_BUNDLE", "0")),
        search="sweep",
    ),
    Parameter(
        "DS4_EXPERT_CACHE_SLOTS", ("12", "16", "18", "20", "22", "24"), "22",
        profiles=("io", "standard", "full"),
        description="mixed expert-cache byte budget (~250 MiB/base slot)",
        memory_risk=True, search="sweep",
    ),
    Parameter(
        "DS4_EXPERT_CACHE_UNIFORM", ("0", "1"), "1",
        profiles=("io", "standard", "full"),
        description="uniform vs usage-driven byte allocation",
        requires_nonzero=("DS4_EXPERT_CACHE_SLOTS",), usage_dependent=True,
    ),
    Parameter(
        "DS4_EXPERT_LOOKAHEAD", ("0", "2", "4", "6", "8", "10", "12"), "0",
        profiles=("io", "standard", "full"),
        description="next-layer usage-prior expert fills",
        requires=(("DS4_MULTI_QUANT_CACHE", "1"),),
        requires_nonzero=("DS4_EXPERT_CACHE_SLOTS",), usage_dependent=True,
        search="sweep",
    ),
    Parameter(
        "DS4_DENSE_AHEAD", ("1", "2", "3"), "2",
        profiles=("io", "standard", "full"),
        description="dense staging-ring read-ahead depth",
        requires=(("DS4_DENSE_STREAM", "1"),),
        memory_risk=True,
    ),
    Parameter(
        "DS4_MOE_NSG", string_values(range(1, 9)), "4",
        profiles=("standard", "full"),
        description="row occupancy for routed MoE kernels",
        search="sweep",
    ),
    Parameter(
        "DS4_DENSE_Q4_NSG", string_values(range(1, 9)), "4",
        profiles=("standard", "full"),
        description="row occupancy for resident dense Q4 kernels",
        requires=(("DS4_DENSE_Q4", "1"), ("DS4_DENSE_Q4_KERNEL", "1")),
        search="sweep",
    ),
    Parameter(
        "DS4_ASYNC_FFN", ("0", "1"), "1",
        profiles=("full",), description="asynchronous routed FFN pipeline",
        requires=(("DS4_PROFILE_ROUTE", "0"),),
    ),
    Parameter(
        "DS4_ASYNC_ROUTE", ("0", "1"), "1",
        profiles=("full",), description="asynchronous route/shared-FFN commit",
        requires=(("DS4_PROFILE_ROUTE", "0"),),
    ),
    Parameter(
        "DS4_MULTI_QUANT_CACHE", ("0", "1"), "1",
        profiles=("full",), description="per-layer mixed expert pool geometry",
        requires_nonzero=("DS4_EXPERT_CACHE_SLOTS",),
    ),
    Parameter(
        "DS4_POOL_INTERLEAVE", ("0", "1"), "1",
        profiles=("full",), description="interleaved gate/up/down cache slots",
        requires_nonzero=("DS4_EXPERT_CACHE_SLOTS",),
    ),
    Parameter(
        "DS4_RESIDENT_COMP", ("0", "1"), "1",
        profiles=("full",), description="resident NSA compressor projections",
        requires=(("DS4_DENSE_STREAM", "1"),), memory_risk=True,
    ),
    Parameter(
        "DS4_MLOCK", ("0", "1"), "1",
        profiles=("full",), description="best-effort pinning of hot buffers",
        memory_risk=True,
    ),
    Parameter(
        "DS4_ADAPTIVE_SPLITK", ("0", "1"), "1",
        profiles=("full",), description="adaptive FlashAttention split-K depth",
    ),
    Parameter(
        "DS4_DENSE_Q4_KERNEL", ("0", "1"), "1",
        profiles=("full",), description="dedicated dense Q4 matvec",
        requires=(("DS4_DENSE_Q4", "1"),),
    ),
    Parameter(
        "DS4_FUSED_ROUTER_FINALIZE", ("0", "1"), "1",
        profiles=("full",), description="fused top-k plus route normalization",
    ),
    Parameter(
        "DS4_FUSED_COMP_PROJ", ("0", "1"), "1",
        profiles=("full",), description="paired compressor KV/gate projection",
    ),
    Parameter(
        "DS4_VECTOR_COPY", ("0", "1"), "0",
        profiles=("full",), description="packed-four contiguous copies",
    ),
    Parameter(
        "DS4_FLASH_KV_STAGE", ("0", "1"), "0",
        profiles=("full",), description="fused FlashAttention KV staging",
    ),
    Parameter(
        "DS4_ROPE_PAIR", ("0", "1"), "0",
        profiles=("full",), description="pair-only in-place RoPE kernel",
    ),
    # Prefill knobs use a separate objective and need a representative prompt.
    Parameter(
        "DS4_PREFILL_UNION", ("128", "160", "192", "224", "256"), "256",
        metric="prefillTps", profiles=("prefill", "full"),
        description="expert union size per layer-major prefill group",
        min_prompt_tokens=1024, memory_risk=True,
    ),
    Parameter(
        "DS4_PREFILL_CHUNK", ("256", "512", "768", "1024"), "512",
        metric="prefillTps", profiles=("prefill", "full"),
        description="tokens per layer-major prefill chunk",
        min_prompt_tokens=1024, min_prompt_equals_value=True, memory_risk=True,
    ),
    Parameter(
        "DS4_PREFILL_ROUTE_BATCH", ("8", "16", "32", "64", "128"), "32",
        metric="prefillTps", profiles=("prefill", "full"),
        description="route/attention tokens per command buffer",
        min_prompt_tokens=128, search="sweep",
    ),
    Parameter(
        "DS4_PREFILL_FFN_BATCH", ("0", "1"), "1",
        metric="prefillTps", profiles=("prefill", "full"),
        description="batched prefill FFN command buffers",
        min_prompt_tokens=128,
    ),
    # Numeric-close knobs are opt-in and use stricter gates than the generic
    # metal_ab comparator.  They are never selected without --allow-numeric.
    Parameter(
        "DS4_Q8_NSG", string_values(range(1, 9)), "4",
        quality="numeric", profiles=("numeric", "full"),
        description="Q8 K-reduction occupancy (last bits may change)",
        search="sweep",
    ),
    Parameter(
        "DS4_FUSED_HC", ("0", "1"), "1",
        quality="numeric", profiles=("numeric", "full"),
        description="fused HyperConnection reduction (±1 ulp class)",
    ),
)

PARAMETER_BY_NAME = {parameter.name: parameter for parameter in PARAMETERS}


@dataclass
class RunResult:
    index: int
    label: str
    workload: str
    directory: str
    log_path: str
    trace_prefix: str
    config: dict[str, str]
    performance: dict[str, float | None]
    diagnostics: dict[str, Any]
    returncode: int
    duration_s: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.error is None


@dataclass
class Trial:
    trial_id: int
    pass_index: int
    parameter: str
    from_value: str
    to_value: str
    workload: str
    metric: str
    quality_mode: str
    run_dirs: list[str] = field(default_factory=list)
    baseline_values: list[float] = field(default_factory=list)
    candidate_values: list[float] = field(default_factory=list)
    ratio_ab: float | None = None
    ratio_ba: float | None = None
    balanced_ratio: float | None = None
    secondary_ratio: float | None = None
    candidate_stability: float | None = None
    candidate_diagnostics: dict[str, float | None] = field(default_factory=dict)
    transition_quality: dict[str, Any] = field(default_factory=dict)
    cumulative_quality: dict[str, Any] = field(default_factory=dict)
    qualified: bool = False
    selected: bool = False
    reason: str = ""
    candidate_config: dict[str, str] = field(default_factory=dict)


class TunerError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def median(values: Iterable[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.median(finite) if finite else None


def geometric_ratio(first: float | None, second: float | None) -> float | None:
    if first is None or second is None or first <= 0 or second <= 0:
        return None
    return math.sqrt(first * second)


def safe_ratio(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline <= 0:
        return None
    return candidate / baseline


def system_memory_snapshot() -> dict[str, float | None]:
    """Read lightweight macOS pressure counters without allocating model-sized data."""
    result: dict[str, float | None] = {
        "freePercent": None,
        "swapoutsPages": None,
        "pageSize": None,
    }
    try:
        output = subprocess.check_output(
            ["/usr/bin/memory_pressure", "-Q"], text=True,
            stderr=subprocess.DEVNULL, timeout=5,
        )
        match = re.search(r"memory free percentage:\s*([0-9]+(?:\.[0-9]+)?)%", output)
        if match:
            result["freePercent"] = float(match.group(1))
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass
    try:
        output = subprocess.check_output(
            ["/usr/bin/vm_stat"], text=True,
            stderr=subprocess.DEVNULL, timeout=5,
        )
        page = re.search(r"page size of\s+(\d+) bytes", output)
        swapouts = re.search(r"^Swapouts:\s+(\d+)\.", output, re.MULTILINE)
        if page:
            result["pageSize"] = float(page.group(1))
        if swapouts:
            result["swapoutsPages"] = float(swapouts.group(1))
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass
    return result


def terminate_process_group(process: subprocess.Popen[bytes], grace_s: float = 10.0) -> None:
    """Terminate a DS4Demo process group and always reap the child."""
    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    else:
        process.wait()


def parse_diagnostics(path: str, warmup: int) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")

    def last_float(pattern: str, flags: int = 0, group: int = 1) -> float | None:
        found = list(re.finditer(pattern, text, flags))
        return float(found[-1].group(group)) if found else None

    def last_int(pattern: str, flags: int = 0, group: int = 1) -> int | None:
        value = last_float(pattern, flags, group)
        return int(value) if value is not None else None

    speeds = [
        float(match.group(1))
        for match in re.finditer(r"\[tok\s+\d+\s+[^\]]*?([0-9]+(?:\.[0-9]+)?) tok/s\]", text)
    ]
    steady = speeds[warmup:] if len(speeds) > warmup else speeds
    stability: float | None = None
    if len(steady) >= 6:
        midpoint = len(steady) // 2
        head = median(steady[:midpoint])
        tail = median(steady[midpoint:])
        if head and tail is not None:
            stability = tail / head

    return {
        "promptTokens": last_int(r"DS4Demo: prompt .*? -> (\d+) tokens; generating", re.S),
        "routeMsPerToken": last_float(
            r"Profilo decode[\s\S]*?route/attn\s+([0-9]+(?:\.[0-9]+)?) ms/token"
        ),
        "gatherMsPerToken": last_float(
            r"Profilo decode[\s\S]*?gather IO\s+([0-9]+(?:\.[0-9]+)?) ms/token"
        ),
        "expertsMsPerToken": last_float(
            r"Profilo decode[\s\S]*?experts\s+([0-9]+(?:\.[0-9]+)?) ms/token"
        ),
        "gatherMBPerToken": last_float(
            r"gather IO\s+([0-9]+(?:\.[0-9]+)?) MB/token"
        ),
        "gatherGBs": last_float(
            r"gather IO\s+[0-9]+(?:\.[0-9]+)? MB/token.*?banda effettiva ([0-9]+(?:\.[0-9]+)?) GB/s"
        ),
        "cacheGlobalByteHitPercent": last_float(
            r"cache byte\s+[0-9]+(?:\.[0-9]+)?% hit sui cacheabili / ([0-9]+(?:\.[0-9]+)?)% globale"
        ),
        "stabilityTailHead": stability,
        "tokenSpeeds": speeds,
    }


def quality_summary(
    reference_prefix: str,
    candidate_prefix: str,
    mode: str,
    *,
    numeric_atol: float = 1e-4,
    numeric_rtol: float = 1e-5,
    numeric_max_abs: float = 1e-3,
    numeric_nrmse: float = 1e-5,
) -> dict[str, Any]:
    if mode not in {"exact", "numeric"}:
        return {
            "passed": False,
            "mode": mode,
            "verdict": "ERROR",
            "error": f"quality mode non supportato: {mode!r}",
        }
    atol = 0.0 if mode == "exact" else numeric_atol
    rtol = 0.0 if mode == "exact" else numeric_rtol
    try:
        comparison = compare_traces(reference_prefix, candidate_prefix, atol, rtol)
        stats = comparison["stats"]
        max_frame_nrmse = max(
            (float(row["nrmse"]) for row in comparison["frameRows"]), default=0.0
        )
        exact_percent = 100.0 * stats.exact / max(stats.count, 1)
        within_percent = 100.0 * stats.within / max(stats.count, 1)
        if mode == "exact":
            passed = comparison["verdict"] == "PASS_EXACT"
        else:
            passed = (
                comparison["verdict"] in {"PASS_EXACT", "PASS_NUMERIC"}
                and comparison["generatedSame"]
                and comparison["argmaxSame"]
                and comparison["top3Same"]
                and not comparison["structuralErrors"]
                and stats.nonfinite == 0
                and stats.within == stats.count
                and stats.max_abs <= numeric_max_abs
                and stats.nrmse <= numeric_nrmse
                and max_frame_nrmse <= numeric_nrmse
            )
        return {
            "passed": passed,
            "mode": mode,
            "verdict": comparison["verdict"],
            "generatedSame": comparison["generatedSame"],
            "argmaxSame": comparison["argmaxSame"],
            "top3Same": comparison["top3Same"],
            "exactPercent": exact_percent,
            "withinPercent": within_percent,
            "maxAbs": stats.max_abs,
            "nrmse": stats.nrmse,
            "maxFrameNrmse": max_frame_nrmse,
            "nonfinite": stats.nonfinite,
            "structuralErrors": comparison["structuralErrors"],
        }
    except Exception as error:  # A malformed/truncated trace is always a reject.
        return {
            "passed": False,
            "mode": mode,
            "verdict": "ERROR",
            "error": f"{type(error).__name__}: {error}",
        }


def ordered_neighbors(values: tuple[str, ...], current: str) -> list[tuple[str, int]]:
    if current not in values:
        return []
    index = values.index(current)
    neighbors: list[tuple[str, int]] = []
    # Prefer the upward direction, but always measure both immediate neighbors.
    if index + 1 < len(values):
        neighbors.append((values[index + 1], +1))
    if index > 0:
        neighbors.append((values[index - 1], -1))
    return neighbors


def shell_exports(environment: dict[str, str]) -> str:
    lines = ["#!/bin/sh", "# Generated by scripts/metal_autotune.py"]
    for key in sorted(environment):
        if key.startswith("DS4_") and key not in TRANSIENT_DS4_KEYS:
            lines.append(f"export {key}={shlex.quote(environment[key])}")
    return "\n".join(lines) + "\n"


class OutputLock:
    def __init__(self, path: Path):
        self.path = path
        self.acquired = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                pid = int(self.path.read_text(encoding="utf-8").strip())
                os.kill(pid, 0)
            except (ValueError, ProcessLookupError):
                self.path.unlink(missing_ok=True)
            except PermissionError:
                raise TunerError(f"lock esistente non ispezionabile (PID {pid}): {self.path}")
            else:
                raise TunerError(f"autotuner già attivo con PID {pid}: {self.path}")
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.close(fd)
        self.acquired = True

    def release(self) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False


class Autotuner:
    def __init__(self, args: argparse.Namespace, parameters: list[Parameter]):
        self.args = args
        self.parameters = parameters
        self.gguf = Path(args.gguf).expanduser().resolve()
        self.decode_prompt = Path(args.prompt).expanduser().resolve()
        self.prefill_prompt = (
            Path(args.prefill_prompt).expanduser().resolve()
            if args.prefill_prompt else self.decode_prompt
        )
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.output = (
            Path(args.output).expanduser().resolve()
            if args.output else (Path(tempfile.gettempdir()) / f"ds4-metal-autotune-{timestamp}")
        )
        self.binary = Path(args.binary).expanduser().resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "runs").mkdir(exist_ok=True)
        (self.output / "inputs").mkdir(exist_ok=True)
        self.lock = OutputLock(self.output / ".autotune.lock")
        self.lock.acquire()
        atexit.register(self.lock.release)

        self.state_path = self.output / "state.json"
        self.events_path = self.output / "events.jsonl"
        self.trials: list[Trial] = []
        self.skipped: list[dict[str, str]] = []
        self.root_runs: dict[str, RunResult] = {}
        self.final_validation: dict[str, Any] = {}
        self.run_counter = 0
        self.trial_counter = 0
        self.completed_passes = 0
        self.status = "initializing"
        self.cumulative_quality_mode = "exact"

        inherited = {
            key: value for key, value in os.environ.items()
            if key.startswith("DS4_") and key not in TRANSIENT_DS4_KEYS
            and not key.startswith("DS4_AB_")
        }
        if args.preset == "inherit":
            missing = [parameter.name for parameter in parameters if parameter.name not in inherited]
            if missing:
                raise TunerError(
                    "--preset inherit richiede un valore esplicito per ogni knob selezionato; "
                    "mancano: " + ", ".join(missing)
                )
        preset = {} if args.preset == "inherit" else dict(M1PRO16_MIXED_PRESET)
        preset.update(inherited)
        if args.context is not None:
            preset["DS4_DEMO_CONTEXT"] = str(args.context)
        self.base_env = preset
        self.initial_config = {
            parameter.name: self.base_env.get(parameter.name, parameter.default)
            for parameter in parameters
        }
        self.current_config = dict(self.initial_config)
        self.initial_config_all = dict(self.base_env)
        self.initial_config_all.update(self.initial_config)
        self.usage_seed = self._prepare_usage_seed(args.usage_seed)

        self.fingerprint = self._fingerprint()
        if args.resume:
            self._load_resume()
        elif self.state_path.exists():
            raise TunerError(
                f"{self.state_path} esiste già; usa --resume oppure scegli un'altra --output"
            )
        self.status = "running"
        self._checkpoint()

    def _prepare_usage_seed(self, requested: str) -> Path | None:
        if requested == "off":
            return None
        destination = self.output / "inputs" / "usage-seed.json"
        if self.args.resume and destination.is_file():
            return destination
        source: Path | None = None
        if requested != "auto":
            candidate = Path(requested).expanduser().resolve()
            if not candidate.is_file():
                raise TunerError(f"usage seed non trovato: {candidate}")
            source = candidate
        else:
            model_name = self.gguf.name
            roots = [
                Path.home() / "Library/Containers/com.dwarfstar.app/Data/Library/Application Support/DwarfStar",
                Path.home() / "Library/Application Support/DwarfStar",
            ]
            candidates: list[Path] = []
            for root in roots:
                if root.is_dir():
                    candidates.extend(root.glob(f"expert-usage-{model_name}-*.json"))
            if candidates:
                source = max(candidates, key=lambda path: path.stat().st_size)
        if source is None:
            return None
        if not destination.exists() or not self.args.resume:
            shutil.copy2(source, destination)
        return destination

    def _fingerprint(self) -> dict[str, Any]:
        def stat(path: Path) -> dict[str, Any]:
            info = path.stat()
            return {"path": str(path), "size": info.st_size, "mtimeNs": info.st_mtime_ns}

        try:
            git = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            git = "unknown"
        manifest = [asdict(parameter) for parameter in self.parameters]
        policy = {
            "minGain": self.args.min_gain,
            "screenRegression": self.args.screen_regression,
            "maxPairRegression": self.args.max_pair_regression,
            "maxSecondaryRegression": self.args.max_secondary_regression,
            "stabilityFloor": self.args.stability_floor,
            "baselineRetries": self.args.baseline_retries,
            "numericAtol": self.args.numeric_atol,
            "numericRtol": self.args.numeric_rtol,
            "numericMaxAbs": self.args.numeric_max_abs,
            "numericNrmse": self.args.numeric_nrmse,
            "minMemoryFreePercent": self.args.min_memory_free_percent,
            "maxSwapoutMiB": self.args.max_swapout_mib,
            "allowNumeric": self.args.allow_numeric,
            "allowColdUsage": self.args.allow_cold_usage,
            "finalValidation": self.args.final_validation,
        }
        script_paths = [Path(__file__).resolve(), Path(__file__).with_name("metal_ab_compare.py")]
        payload = {
            "schema": SCHEMA,
            "git": git,
            "gguf": stat(self.gguf),
            "decodePrompt": {**stat(self.decode_prompt), "sha256": sha256_file(self.decode_prompt)},
            "prefillPrompt": {**stat(self.prefill_prompt), "sha256": sha256_file(self.prefill_prompt)},
            "binary": stat(self.binary),
            "usageSeed": (
                {**stat(self.usage_seed), "sha256": sha256_file(self.usage_seed)}
                if self.usage_seed else None
            ),
            "baseEnv": self.base_env,
            "manifest": manifest,
            "scripts": {str(path): sha256_file(path) for path in script_paths},
            "policy": policy,
            "maxNew": self.args.max_new,
            "warmup": self.args.warmup,
            "traceFrames": self.args.trace_frames,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return {"sha256": hashlib.sha256(encoded).hexdigest(), "details": payload}

    def _load_resume(self) -> None:
        if not self.state_path.is_file():
            raise TunerError(f"checkpoint non trovato per --resume: {self.state_path}")
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if state.get("fingerprint", {}).get("sha256") != self.fingerprint["sha256"]:
            raise TunerError(
                "fingerprint diverso: binario, modello, prompt, ambiente o manifest sono cambiati"
            )
        if state.get("status") in {"complete", "complete_unvalidated"} and not self.args.continue_complete:
            raise TunerError("autotune già completato; usa --continue-complete per una nuova passata")
        self.base_env = {str(k): str(v) for k, v in state["baseEnv"].items()}
        self.initial_config = {str(k): str(v) for k, v in state["initialConfig"].items()}
        self.current_config = {str(k): str(v) for k, v in state["currentConfig"].items()}
        self.initial_config_all = {str(k): str(v) for k, v in state["initialConfigAll"].items()}
        self.cumulative_quality_mode = state.get("cumulativeQualityMode", "exact")
        self.run_counter = int(state.get("runCounter", 0))
        # A hard kill can leave a reserved/incomplete run directory newer than
        # state.json. Never reuse that identifier on resume.
        for path in (self.output / "runs").glob("[0-9][0-9][0-9][0-9]-*"):
            try:
                self.run_counter = max(self.run_counter, int(path.name.split("-", 1)[0]))
            except ValueError:
                continue
        self.trial_counter = int(state.get("trialCounter", 0))
        self.completed_passes = int(state.get("completedPasses", 0))
        self.trials = [Trial(**trial) for trial in state.get("trials", [])]
        self.skipped = list(state.get("skipped", []))
        self.root_runs = {
            name: RunResult(**run) for name, run in state.get("rootRuns", {}).items()
        }
        self.final_validation = dict(state.get("finalValidation", {}))

    def _state(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "status": self.status,
            "fingerprint": self.fingerprint,
            "baseEnv": self.base_env,
            "initialConfig": self.initial_config,
            "initialConfigAll": self.initial_config_all,
            "currentConfig": self.current_config,
            "cumulativeQualityMode": self.cumulative_quality_mode,
            "runCounter": self.run_counter,
            "trialCounter": self.trial_counter,
            "completedPasses": self.completed_passes,
            "parameters": [asdict(parameter) for parameter in self.parameters],
            "trials": [asdict(trial) for trial in self.trials],
            "skipped": self.skipped,
            "rootRuns": {name: asdict(run) for name, run in self.root_runs.items()},
            "finalValidation": self.final_validation,
            "usageSeed": str(self.usage_seed) if self.usage_seed else None,
        }

    def _checkpoint(self) -> None:
        atomic_write_json(self.state_path, self._state())
        self._write_outputs()

    def _event(self, kind: str, value: dict[str, Any]) -> None:
        payload = {"time": datetime.now().isoformat(), "kind": kind, **value}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _effective_env(self, config: dict[str, str], run_dir: Path, trace_prefix: Path) -> dict[str, str]:
        environment = dict(os.environ)
        for key in list(environment):
            if key.startswith("DS4_"):
                del environment[key]
        environment.update(self.base_env)
        environment.update(config)
        environment.update({
            "DS4_AB_TRACE": str(trace_prefix),
            "DS4_AB_TRACE_FRAMES": str(self.args.trace_frames),
            "DS4_DIAG": "0",
            "DS4_WARMUP": str(self.args.warmup),
            "DS4_DEMO_TEMPERATURE": "0",
            "DS4_DEMO_TOP_K": "0",
            "DS4_DEMO_TOP_P": "1",
            "DS4_DEMO_MIN_P": "0",
            "DS4_DEMO_REPEAT_PENALTY": "1",
            "DS4_SPEC_K": "0",
        })
        if self.usage_seed:
            usage_copy = run_dir / "usage.json"
            shutil.copy2(self.usage_seed, usage_copy)
            environment["DS4_USAGE_FILE"] = str(usage_copy)
        else:
            environment["DS4_USAGE_FILE"] = "off"
        return environment

    def _prompt_for(self, workload: str) -> Path:
        return self.prefill_prompt if workload == "prefill" else self.decode_prompt

    def _run(self, config: dict[str, str], workload: str, label: str) -> RunResult:
        if not self.args.allow_active_download:
            active = active_downloads_near(self.gguf)
            if active:
                raise TunerError(
                    "download parziale recente nella directory modelli; autotune sospeso: "
                    + ", ".join(str(path) for path in active)
                )
        self.run_counter += 1
        # Persist the reservation before touching the run directory/process so
        # kill -9 and power loss can only leave a harmless gap, never an ID clash.
        self._checkpoint()
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-")[:80]
        run_dir = self.output / "runs" / f"{self.run_counter:04d}-{safe_label}"
        run_dir.mkdir(parents=True, exist_ok=False)
        log_path = run_dir / "run.log"
        trace_prefix = run_dir / "trace"
        environment = self._effective_env(config, run_dir, trace_prefix)
        atomic_write_json(
            run_dir / "environment.json",
            {key: value for key, value in environment.items() if key.startswith("DS4_")},
        )
        prompt = self._prompt_for(workload)
        command = [str(self.binary), str(self.gguf), str(self.args.max_new), "@" + str(prompt)]
        print(f"  [run {self.run_counter:04d}] {label} ({workload})")
        if self.args.cooldown > 0 and self.run_counter > 1:
            time.sleep(self.args.cooldown)
        memory_before = system_memory_snapshot()
        started = time.monotonic()
        returncode = -1
        error: str | None = None
        process: subprocess.Popen[bytes] | None = None
        with log_path.open("wb") as log:
            try:
                process = subprocess.Popen(
                    command, cwd=ROOT, env=environment, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True,
                )
                returncode = process.wait(timeout=self.args.timeout)
            except subprocess.TimeoutExpired:
                error = f"timeout dopo {self.args.timeout}s"
                if process is not None:
                    terminate_process_group(process)
                    returncode = process.returncode if process.returncode is not None else -1
            except KeyboardInterrupt:
                if process is not None:
                    terminate_process_group(process)
                raise
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if process is not None:
                    terminate_process_group(process)
                if process is not None and process.returncode is not None:
                    returncode = process.returncode
        duration = time.monotonic() - started
        memory_after = system_memory_snapshot()
        performance = parse_performance(str(log_path)) if log_path.exists() else {}
        diagnostics = parse_diagnostics(str(log_path), self.args.warmup) if log_path.exists() else {}
        diagnostics["memoryFreeBeforePercent"] = memory_before.get("freePercent")
        diagnostics["memoryFreeAfterPercent"] = memory_after.get("freePercent")
        before_swap = memory_before.get("swapoutsPages")
        after_swap = memory_after.get("swapoutsPages")
        page_size = memory_after.get("pageSize") or memory_before.get("pageSize")
        diagnostics["swapoutDeltaMiB"] = (
            max(0.0, after_swap - before_swap) * page_size / (1024.0 * 1024.0)
            if before_swap is not None and after_swap is not None and page_size is not None
            else None
        )
        if returncode != 0 and error is None:
            error = f"DS4Demo exit {returncode}"
        if not Path(str(trace_prefix) + ".json").is_file() or not Path(str(trace_prefix) + ".f32").is_file():
            if error is None:
                error = "trace A/B mancante"
        result = RunResult(
            index=self.run_counter, label=label, workload=workload,
            directory=str(run_dir), log_path=str(log_path), trace_prefix=str(trace_prefix),
            config=dict(config), performance=performance, diagnostics=diagnostics,
            returncode=returncode, duration_s=duration, error=error,
        )
        self._event("run", asdict(result))
        self._checkpoint()
        return result

    def _metric(self, run: RunResult, metric: str) -> float | None:
        value = run.performance.get(metric)
        if value is not None:
            return float(value)
        if metric == "decodeSteadyTps":
            for fallback in ("decodeProfileTps", "decodeWallTps"):
                value = run.performance.get(fallback)
                if value is not None:
                    return float(value)
        return None

    @staticmethod
    def _secondary_metric(metric: str) -> str:
        return "decodeSteadyTps" if metric == "prefillTps" else "prefillTps"

    def _is_stable(self, run: RunResult) -> bool:
        stability = run.diagnostics.get("stabilityTailHead")
        return (
            (stability is None or float(stability) >= self.args.stability_floor)
            and self._memory_issue(run) is None
        )

    def _memory_issue(self, run: RunResult) -> str | None:
        free = run.diagnostics.get("memoryFreeAfterPercent")
        swapout = run.diagnostics.get("swapoutDeltaMiB")
        if free is not None and float(free) < self.args.min_memory_free_percent:
            return (
                f"pressione RAM: memoria libera {float(free):.1f}% "
                f"< {self.args.min_memory_free_percent:.1f}%"
            )
        if swapout is not None and float(swapout) > self.args.max_swapout_mib:
            return (
                f"swapout durante il run {float(swapout):.1f} MiB "
                f"> {self.args.max_swapout_mib:.1f} MiB"
            )
        return None

    def _stable_baseline(self, config: dict[str, str], workload: str, label: str) -> RunResult:
        attempts = 0
        while True:
            run = self._run(config, workload, label + (f"-retry{attempts}" if attempts else ""))
            if run.ok and self._metric(run, "decodeSteadyTps") is not None and self._is_stable(run):
                return run
            attempts += 1
            if attempts > self.args.baseline_retries:
                raise TunerError(
                    f"baseline instabile/fallita dopo {attempts} tentativi: {run.log_path} "
                    f"({run.error or self._memory_issue(run) or 'stabilità coda/testa insufficiente'})"
                )

    def _workload_for(self, parameter: Parameter) -> str:
        if parameter.metric == "prefillTps" and self.prefill_prompt != self.decode_prompt:
            return "prefill"
        return "decode"

    def _root_for(self, workload: str) -> RunResult:
        if workload not in self.root_runs:
            root = self._stable_baseline(
                self.initial_config, workload, f"quality-root-{workload}"
            )
            self.root_runs[workload] = root
            self._checkpoint()
        return self.root_runs[workload]

    def _quality_pair(
        self, reference: RunResult, candidate: RunResult, mode: str
    ) -> dict[str, Any]:
        if not reference.ok or not candidate.ok:
            return {
                "passed": False, "mode": mode, "verdict": "RUN_ERROR",
                "referenceError": reference.error, "candidateError": candidate.error,
            }
        return quality_summary(
            reference.trace_prefix, candidate.trace_prefix, mode,
            numeric_atol=self.args.numeric_atol,
            numeric_rtol=self.args.numeric_rtol,
            numeric_max_abs=self.args.numeric_max_abs,
            numeric_nrmse=self.args.numeric_nrmse,
        )

    def _merge_quality(self, summaries: list[dict[str, Any]], mode: str) -> dict[str, Any]:
        if not summaries:
            return {"passed": False, "mode": mode, "verdict": "MISSING"}
        passed = all(bool(summary.get("passed")) for summary in summaries)
        verdicts = {str(summary.get("verdict")) for summary in summaries}
        if not passed:
            verdict = "FAIL"
        elif verdicts == {"PASS_EXACT"}:
            verdict = "PASS_EXACT"
        else:
            verdict = "PASS_NUMERIC"
        return {
            "passed": passed,
            "mode": mode,
            "verdict": verdict,
            "samples": summaries,
            "worstMaxAbs": max((float(s.get("maxAbs", 0)) for s in summaries), default=0),
            "worstNrmse": max((float(s.get("nrmse", 0)) for s in summaries), default=0),
            "minExactPercent": min((float(s.get("exactPercent", 0)) for s in summaries), default=0),
        }

    def _cleanup_traces(self, runs: Iterable[RunResult]) -> None:
        if self.args.keep_traces:
            return
        protected = {run.trace_prefix for run in self.root_runs.values()}
        for run in runs:
            if run.trace_prefix in protected:
                continue
            Path(run.trace_prefix + ".json").unlink(missing_ok=True)
            Path(run.trace_prefix + ".f32").unlink(missing_ok=True)

    def _evaluate(
        self,
        parameter: Parameter,
        candidate_config: dict[str, str],
        pass_index: int,
    ) -> Trial:
        self.trial_counter += 1
        workload = self._workload_for(parameter)
        root = self._root_for(workload)
        from_value = self.current_config[parameter.name]
        to_value = candidate_config[parameter.name]
        trial = Trial(
            trial_id=self.trial_counter, pass_index=pass_index,
            parameter=parameter.name, from_value=from_value, to_value=to_value,
            workload=workload, metric=parameter.metric, quality_mode=parameter.quality,
            candidate_config=dict(candidate_config),
        )
        print(f"\n[trial {trial.trial_id}] {parameter.name}: {from_value} -> {to_value}")
        a1 = self._stable_baseline(
            self.current_config, workload,
            f"t{trial.trial_id}-{parameter.name}-{from_value}-A1",
        )
        c1 = self._run(
            candidate_config, workload,
            f"t{trial.trial_id}-{parameter.name}-{to_value}-C1",
        )
        runs = [a1, c1]
        trial.run_dirs = [run.directory for run in runs]
        if not c1.ok:
            trial.reason = f"candidato fallito: {c1.error}"
            self.trials.append(trial)
            self._cleanup_traces(runs)
            self._event("trial", asdict(trial)); self._checkpoint()
            return trial
        memory_issue = self._memory_issue(c1)
        if memory_issue:
            trial.reason = memory_issue
            self.trials.append(trial)
            self._cleanup_traces(runs)
            self._event("trial", asdict(trial)); self._checkpoint()
            return trial

        transition_mode = parameter.quality
        cumulative_mode = (
            "numeric" if parameter.quality == "numeric"
            or self.cumulative_quality_mode == "numeric" else "exact"
        )
        transition = [self._quality_pair(a1, c1, transition_mode)]
        cumulative = [self._quality_pair(root, c1, cumulative_mode)]
        trial.transition_quality = self._merge_quality(transition, transition_mode)
        trial.cumulative_quality = self._merge_quality(cumulative, cumulative_mode)
        if not trial.transition_quality["passed"] or not trial.cumulative_quality["passed"]:
            trial.reason = "regressione qualità nello screening"
            self.trials.append(trial)
            self._cleanup_traces(runs)
            self._event("trial", asdict(trial)); self._checkpoint()
            return trial

        a1_value = self._metric(a1, parameter.metric)
        c1_value = self._metric(c1, parameter.metric)
        ratio_ab = safe_ratio(c1_value, a1_value)
        trial.ratio_ab = ratio_ab
        if a1_value is not None:
            trial.baseline_values.append(a1_value)
        if c1_value is not None:
            trial.candidate_values.append(c1_value)
        if ratio_ab is None:
            trial.reason = f"metrica {parameter.metric} assente"
            self.trials.append(trial)
            self._cleanup_traces(runs)
            self._event("trial", asdict(trial)); self._checkpoint()
            return trial
        if ratio_ab < 1.0 - self.args.screen_regression:
            trial.reason = f"screening lento: {(ratio_ab - 1) * 100:+.1f}%"
            self.trials.append(trial)
            self._cleanup_traces(runs)
            self._event("trial", asdict(trial)); self._checkpoint()
            return trial

        # Reverse order: C2 then A2. Together with A1 then C1 this is ABBA.
        c2 = self._run(
            candidate_config, workload,
            f"t{trial.trial_id}-{parameter.name}-{to_value}-C2",
        )
        a2 = self._stable_baseline(
            self.current_config, workload,
            f"t{trial.trial_id}-{parameter.name}-{from_value}-A2",
        )
        runs.extend([c2, a2])
        trial.run_dirs = [run.directory for run in runs]
        if not c2.ok:
            trial.reason = f"conferma candidato fallita: {c2.error}"
            self.trials.append(trial)
            self._cleanup_traces(runs)
            self._event("trial", asdict(trial)); self._checkpoint()
            return trial
        memory_issue = self._memory_issue(c2)
        if memory_issue:
            trial.reason = memory_issue
            self.trials.append(trial)
            self._cleanup_traces(runs)
            self._event("trial", asdict(trial)); self._checkpoint()
            return trial

        transition.append(self._quality_pair(a2, c2, transition_mode))
        cumulative.append(self._quality_pair(root, c2, cumulative_mode))
        trial.transition_quality = self._merge_quality(transition, transition_mode)
        trial.cumulative_quality = self._merge_quality(cumulative, cumulative_mode)
        if not trial.transition_quality["passed"] or not trial.cumulative_quality["passed"]:
            trial.reason = "regressione qualità nella conferma ABBA"
            self.trials.append(trial)
            self._cleanup_traces(runs)
            self._event("trial", asdict(trial)); self._checkpoint()
            return trial

        a2_value = self._metric(a2, parameter.metric)
        c2_value = self._metric(c2, parameter.metric)
        ratio_ba = safe_ratio(c2_value, a2_value)
        balanced = geometric_ratio(ratio_ab, ratio_ba)
        trial.ratio_ba = ratio_ba
        trial.balanced_ratio = balanced
        if a2_value is not None:
            trial.baseline_values.append(a2_value)
        if c2_value is not None:
            trial.candidate_values.append(c2_value)

        secondary = self._secondary_metric(parameter.metric)
        secondary_ab = safe_ratio(self._metric(c1, secondary), self._metric(a1, secondary))
        secondary_ba = safe_ratio(self._metric(c2, secondary), self._metric(a2, secondary))
        trial.secondary_ratio = geometric_ratio(secondary_ab, secondary_ba)
        stabilities = [
            float(run.diagnostics["stabilityTailHead"])
            for run in (c1, c2) if run.diagnostics.get("stabilityTailHead") is not None
        ]
        trial.candidate_stability = min(stabilities) if stabilities else None
        diagnostic_keys = (
            "routeMsPerToken", "gatherMsPerToken", "expertsMsPerToken",
            "gatherMBPerToken", "gatherGBs", "cacheGlobalByteHitPercent",
            "memoryFreeAfterPercent", "swapoutDeltaMiB",
        )
        trial.candidate_diagnostics = {
            key: median(
                float(run.diagnostics[key]) for run in (c1, c2)
                if run.diagnostics.get(key) is not None
            )
            for key in diagnostic_keys
        }

        reasons: list[str] = []
        if balanced is None:
            reasons.append("metrica ABBA incompleta")
        elif balanced < 1.0 + self.args.min_gain:
            reasons.append(f"guadagno bilanciato {(balanced - 1) * 100:+.1f}% < {self.args.min_gain * 100:.1f}%")
        if ratio_ab is not None and ratio_ab < 1.0 - self.args.max_pair_regression:
            reasons.append(f"coppia AB regressiva {(ratio_ab - 1) * 100:+.1f}%")
        if ratio_ba is not None and ratio_ba < 1.0 - self.args.max_pair_regression:
            reasons.append(f"coppia BA regressiva {(ratio_ba - 1) * 100:+.1f}%")
        if trial.secondary_ratio is not None and trial.secondary_ratio < 1.0 - self.args.max_secondary_regression:
            reasons.append(
                f"metrica secondaria regressiva {(trial.secondary_ratio - 1) * 100:+.1f}%"
            )
        if trial.candidate_stability is not None and trial.candidate_stability < self.args.stability_floor:
            reasons.append(
                f"collasso progressivo coda/testa {trial.candidate_stability:.2f}"
            )
        trial.qualified = not reasons
        trial.reason = "qualificato" if trial.qualified else "; ".join(reasons)
        self.trials.append(trial)
        self._cleanup_traces(runs)
        self._event("trial", asdict(trial)); self._checkpoint()
        verdict = "QUALIFICATO" if trial.qualified else "respinto"
        gain = "n/d" if balanced is None else f"{(balanced - 1) * 100:+.1f}%"
        print(f"  -> {verdict}: ABBA {gain}; {trial.reason}")
        return trial

    def _skip_reason(self, parameter: Parameter, workload_tokens: int | None) -> str | None:
        effective = dict(self.base_env); effective.update(self.current_config)
        for key, expected in parameter.requires:
            if effective.get(key) != expected:
                return f"richiede {key}={expected} (attuale {effective.get(key, 'unset')})"
        for key in parameter.requires_nonzero:
            try:
                active = int(effective.get(key, "0")) > 0
            except ValueError:
                active = False
            if not active:
                return f"richiede {key}>0 (attuale {effective.get(key, 'unset')})"
        if parameter.usage_dependent and self.usage_seed is None and not self.args.allow_cold_usage:
            return "richiede usage seed maturo (--usage-seed auto|PATH o --allow-cold-usage)"
        if parameter.min_prompt_tokens and (
            workload_tokens is None or workload_tokens < parameter.min_prompt_tokens
        ):
            return (
                f"prompt troppo corto ({workload_tokens or 0} token; "
                f"richiesti >= {parameter.min_prompt_tokens})"
            )
        if self.current_config.get(parameter.name) not in parameter.values:
            return (
                f"valore iniziale {self.current_config.get(parameter.name)!r} fuori dal manifest "
                f"{list(parameter.values)}"
            )
        return None

    @staticmethod
    def _candidate_skip_reason(
        parameter: Parameter, value: str, workload_tokens: int | None
    ) -> str | None:
        if parameter.min_prompt_equals_value:
            required = int(value)
            if workload_tokens is None or workload_tokens < required:
                return f"prompt {workload_tokens or 0} token < valore candidato {required}"
        return None

    def tune(self) -> None:
        # Root decode validates the baseline before any expensive search.
        decode_root = self._root_for("decode")
        print(f"Baseline valida: {decode_root.log_path}")
        if self.usage_seed:
            print(f"Usage seed congelato: {self.usage_seed}")
        else:
            print("Usage seed: OFF (i knob usage-driven saranno saltati)")

        for pass_index in range(self.completed_passes + 1, self.args.max_passes + 1):
            print(f"\n===== PASSATA COORDINATE {pass_index}/{self.args.max_passes} =====")
            pass_changed = False
            for parameter in self.parameters:
                workload = self._workload_for(parameter)
                root = self._root_for(workload)
                tokens = root.diagnostics.get("promptTokens")
                reason = self._skip_reason(parameter, int(tokens) if tokens is not None else None)
                if reason:
                    item = {"pass": str(pass_index), "parameter": parameter.name, "reason": reason}
                    if item not in self.skipped:
                        self.skipped.append(item)
                    print(f"\n[skip] {parameter.name}: {reason}")
                    self._checkpoint()
                    continue

                current = self.current_config[parameter.name]
                if parameter.search == "sweep":
                    candidate_specs = [(value, 0) for value in parameter.values if value != current]
                else:
                    candidate_specs = ordered_neighbors(parameter.values, current)
                if not candidate_specs:
                    continue
                first_trials: list[tuple[Trial, int]] = []
                for value, direction in candidate_specs:
                    candidate_reason = self._candidate_skip_reason(
                        parameter, value, int(tokens) if tokens is not None else None
                    )
                    if candidate_reason:
                        item = {
                            "pass": str(pass_index),
                            "parameter": f"{parameter.name}={value}",
                            "reason": candidate_reason,
                        }
                        if item not in self.skipped:
                            self.skipped.append(item)
                        print(f"\n[skip] {parameter.name}={value}: {candidate_reason}")
                        continue
                    candidate = dict(self.current_config); candidate[parameter.name] = value
                    first_trials.append((self._evaluate(parameter, candidate, pass_index), direction))
                qualified = [(trial, direction) for trial, direction in first_trials if trial.qualified]
                if not qualified:
                    continue
                best, direction = max(
                    qualified, key=lambda item: item[0].balanced_ratio or 0.0
                )
                for trial, _ in qualified:
                    if trial is not best:
                        trial.reason = (
                            "qualificato ma non migliore nella griglia"
                            if parameter.search == "sweep"
                            else "qualificato ma non migliore fra i due vicini"
                        )
                best.selected = True
                best.reason = "promosso"
                self.current_config = dict(best.candidate_config)
                if parameter.quality == "numeric":
                    self.cumulative_quality_mode = "numeric"
                pass_changed = True
                print(
                    f"  PROMOSSO {parameter.name}={self.current_config[parameter.name]} "
                    f"({((best.balanced_ratio or 1) - 1) * 100:+.1f}%)"
                )
                self._event("promotion", asdict(best)); self._checkpoint()

                if parameter.search == "sweep":
                    continue

                # Keep walking only in the winning direction until the first
                # quality/performance failure or the manifest boundary.
                while True:
                    index = parameter.values.index(self.current_config[parameter.name])
                    next_index = index + direction
                    if next_index < 0 or next_index >= len(parameter.values):
                        break
                    candidate = dict(self.current_config)
                    candidate[parameter.name] = parameter.values[next_index]
                    trial = self._evaluate(parameter, candidate, pass_index)
                    if not trial.qualified:
                        break
                    trial.selected = True
                    trial.reason = "promosso lungo la direzione vincente"
                    self.current_config = dict(trial.candidate_config)
                    if parameter.quality == "numeric":
                        self.cumulative_quality_mode = "numeric"
                    print(
                        f"  PROMOSSO {parameter.name}={self.current_config[parameter.name]} "
                        f"({((trial.balanced_ratio or 1) - 1) * 100:+.1f}%)"
                    )
                    self._event("promotion", asdict(trial)); self._checkpoint()

            self.completed_passes = pass_index
            self._checkpoint()
            if not pass_changed:
                print("Nessuna promozione nella passata: convergenza raggiunta.")
                break

        if self.args.final_validation:
            self._validate_final()
        self.status = "complete" if self.args.final_validation else "complete_unvalidated"
        self._checkpoint()
        print(f"\nAutotune completato. Report: {self.output / 'report.md'}")
        print(f"Ambiente migliore: {self.output / 'final-env.sh'}")

    def _validate_final(self) -> None:
        validation: dict[str, Any] = {}
        workloads = {self._workload_for(parameter) for parameter in self.parameters}
        for workload in sorted(workloads):
            root = self._root_for(workload)
            print(f"\n[final validation] {workload}: configurazione iniziale vs finale")
            a1 = self._stable_baseline(self.initial_config, workload, f"final-{workload}-A1")
            c1 = self._run(self.current_config, workload, f"final-{workload}-C1")
            c2 = self._run(self.current_config, workload, f"final-{workload}-C2")
            a2 = self._stable_baseline(self.initial_config, workload, f"final-{workload}-A2")
            metric = "prefillTps" if workload == "prefill" else "decodeSteadyTps"
            ratios = [
                safe_ratio(self._metric(c1, metric), self._metric(a1, metric)),
                safe_ratio(self._metric(c2, metric), self._metric(a2, metric)),
            ]
            quality = self._merge_quality([
                self._quality_pair(root, c1, self.cumulative_quality_mode),
                self._quality_pair(root, c2, self.cumulative_quality_mode),
            ], self.cumulative_quality_mode)
            balanced = geometric_ratio(ratios[0], ratios[1])
            net_changed = any(
                self._workload_for(parameter) == workload
                and self.current_config.get(parameter.name) != self.initial_config.get(parameter.name)
                for parameter in self.parameters
            )
            pair_passed = all(
                ratio is not None and ratio >= 1.0 - self.args.max_pair_regression
                for ratio in ratios
            )
            stability_passed = self._is_stable(c1) and self._is_stable(c2)
            if balanced is None:
                performance_passed = False
            elif net_changed:
                # Promotions must still produce a measurable end-to-end gain
                # when composed. Use half the per-step margin to tolerate the
                # final ABBA's independent noise, but never accept a slowdown.
                performance_passed = balanced >= 1.0 + self.args.min_gain / 2
            else:
                # Knobs selected by the other workload may interact; preserve
                # the same secondary-regression budget used for every trial.
                performance_passed = balanced >= 1.0 - self.args.max_secondary_regression
            performance_passed = performance_passed and pair_passed and stability_passed
            validation[workload] = {
                "metric": metric,
                "ratioAB": ratios[0], "ratioBA": ratios[1],
                "balancedRatio": balanced,
                "netChanged": net_changed,
                "pairPassed": pair_passed,
                "stabilityPassed": stability_passed,
                "performancePassed": performance_passed,
                "quality": quality,
                "runDirs": [run.directory for run in (a1, c1, c2, a2)],
            }
            self._cleanup_traces((a1, c1, c2, a2))
        self.final_validation = validation
        if any(
            not item["quality"].get("passed") or not item.get("performancePassed")
            for item in validation.values()
        ):
            self.status = "validation_failed"
            self._checkpoint()
            raise TunerError(
                "validazione finale qualità/prestazioni fallita; final-env.sh NON va promosso"
            )
        self._event("final_validation", validation)
        self._checkpoint()

    def _write_outputs(self) -> None:
        effective = dict(self.base_env); effective.update(self.current_config)
        validated = (
            self.status == "complete"
            and self.args.final_validation
            and bool(self.final_validation)
            and all(
                item.get("performancePassed") and item.get("quality", {}).get("passed")
                for item in self.final_validation.values()
            )
        )
        warning = (
            "# VALIDATED: quality/performance final validation passed\n"
            if validated
            else f"# NOT FINAL: autotuner status={self.status}\n"
        )
        exports = shell_exports(effective)
        if self.usage_seed:
            exports += f"export DS4_USAGE_FILE={shlex.quote(str(self.usage_seed))}\n"
        first, remainder = exports.split("\n", 1)
        atomic_write_text(self.output / "final-env.sh", first + "\n" + warning + remainder)
        atomic_write_json(self.output / "results.json", self._state())

        rows: list[dict[str, Any]] = []
        for trial in self.trials:
            rows.append({
                "trial": trial.trial_id, "pass": trial.pass_index,
                "parameter": trial.parameter, "from": trial.from_value, "to": trial.to_value,
                "workload": trial.workload, "metric": trial.metric,
                "ratio_ab": trial.ratio_ab, "ratio_ba": trial.ratio_ba,
                "balanced_ratio": trial.balanced_ratio,
                "secondary_ratio": trial.secondary_ratio,
                "baseline_tps": median(trial.baseline_values),
                "candidate_tps": median(trial.candidate_values),
                "candidate_stability": trial.candidate_stability,
                "gather_ms_token": trial.candidate_diagnostics.get("gatherMsPerToken"),
                "gather_mb_token": trial.candidate_diagnostics.get("gatherMBPerToken"),
                "gather_gbs": trial.candidate_diagnostics.get("gatherGBs"),
                "memory_free_percent": trial.candidate_diagnostics.get("memoryFreeAfterPercent"),
                "swapout_mib": trial.candidate_diagnostics.get("swapoutDeltaMiB"),
                "quality": trial.transition_quality.get("verdict"),
                "cumulative_quality": trial.cumulative_quality.get("verdict"),
                "qualified": trial.qualified, "selected": trial.selected,
                "reason": trial.reason,
            })
        csv_path = self.output / "results.csv"
        if rows:
            fd, temporary = tempfile.mkstemp(prefix=csv_path.name + ".", dir=csv_path.parent)
            try:
                with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)
                    handle.flush(); os.fsync(handle.fileno())
                os.replace(temporary, csv_path)
            finally:
                if os.path.exists(temporary): os.unlink(temporary)
        else:
            atomic_write_text(csv_path, "")
        atomic_write_text(self.output / "report.md", self._markdown_report())

    def _markdown_report(self) -> str:
        lines = [
            "# DS4 Metal multi-parameter autotune",
            "",
            f"- Stato: **{self.status}**",
            f"- Modello: `{self.gguf}`",
            f"- Prompt decode: `{self.decode_prompt}`",
            f"- Prompt prefill: `{self.prefill_prompt}`",
            f"- Fingerprint: `{self.fingerprint['sha256']}`",
            f"- Usage seed: `{self.usage_seed if self.usage_seed else 'off'}`",
            f"- Passate completate: `{self.completed_passes}`",
            f"- Gate cumulativo: `{self.cumulative_quality_mode}`",
            "",
            "La ricerca è coordinate-ascent multi-pass, non un prodotto cartesiano. "
            "Le griglie hardware non monotone sono esplorate interamente; gli altri knob "
            "seguono entrambe le direzioni finché migliorano. Ogni finalista usa ordine "
            "ABBA; i knob lossless richiedono logits bit-identici.",
            "",
            "**Nota qualità:** il gate confronta le ottimizzazioni con la configurazione "
            "iniziale del preset, che include quantizzazioni Q4 già abilitate; non confronta "
            "la baseline con il GGUF non quantizzato.",
            "",
            "## Configurazione",
            "",
            "| Parametro | Iniziale | Migliore corrente | RAM-sensitive |",
            "|---|---:|---:|:---:|",
        ]
        for parameter in self.parameters:
            lines.append(
                f"| `{parameter.name}` | `{self.initial_config.get(parameter.name)}` | "
                f"`{self.current_config.get(parameter.name)}` | "
                f"{'sì' if parameter.memory_risk else 'no'} |"
            )
        lines.extend([
            "",
            "## Prove",
            "",
            "| # | Pass | Parametro | A→B | Base | Cand. | ABBA | Secondaria | Stabilità | Gather GB/s | Qualità | Esito |",
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ])
        for trial in self.trials:
            def ratio(value: float | None) -> str:
                return "—" if value is None else f"{(value - 1) * 100:+.1f}%"
            quality = trial.transition_quality.get("verdict", "—")
            result = "PROMOSSO" if trial.selected else "qualificato" if trial.qualified else "respinto"
            base = median(trial.baseline_values)
            candidate = median(trial.candidate_values)
            stable = trial.candidate_stability
            gather = trial.candidate_diagnostics.get("gatherGBs")
            lines.append(
                f"| {trial.trial_id} | {trial.pass_index} | `{trial.parameter}` | "
                f"`{trial.from_value}→{trial.to_value}` | "
                f"{'—' if base is None else f'{base:.3f}'} | "
                f"{'—' if candidate is None else f'{candidate:.3f}'} | "
                f"{ratio(trial.balanced_ratio)} | {ratio(trial.secondary_ratio)} | "
                f"{'—' if stable is None else f'{stable:.2f}'} | "
                f"{'—' if gather is None else f'{float(gather):.2f}'} | "
                f"{quality} | {result}: {trial.reason} |"
            )
        if not self.trials:
            lines.append("| — | — | — | — | — | — | — | — | — | — | — | nessuna prova |")
        if self.skipped:
            lines.extend(["", "## Parametri saltati", ""])
            lines.extend(
                f"- pass {item['pass']}: `{item['parameter']}` — {item['reason']}"
                for item in self.skipped
            )
        if self.final_validation:
            lines.extend(["", "## Validazione finale", ""])
            for workload, value in self.final_validation.items():
                balanced = value.get("balancedRatio")
                gain = "—" if balanced is None else f"{(balanced - 1) * 100:+.1f}%"
                lines.append(
                    f"- **{workload}** `{value['metric']}`: {gain}, "
                    f"qualità `{value['quality'].get('verdict')}`, "
                    f"coppie `{'PASS' if value.get('pairPassed') else 'FAIL'}`, "
                    f"stabilità `{'PASS' if value.get('stabilityPassed') else 'FAIL'}`, "
                    f"prestazioni `{'PASS' if value.get('performancePassed') else 'FAIL'}`."
                )
        lines.extend([
            "",
            "## Artefatti",
            "",
            "- `final-env.sh`: ambiente DS4 sourceable con i valori migliori.",
            "- `results.csv` / `results.json`: risultati macchina-legibili.",
            "- `events.jsonl` / `state.json`: journal e checkpoint per `--resume`.",
            "- `runs/*/run.log`: log completi di ogni processo.",
            "",
            "## Limiti del controllo qualità",
            "",
            "La parità viene verificata sui logits completi dei prompt forniti. "
            "Non equivale a un benchmark semantico su un corpus. Per questo le "
            "quantizzazioni derivate, `DS4_ACTIVE_EXPERTS` e gli altri cambiamenti "
            "lossy non sono inclusi nel tuner automatico.",
            "",
        ])
        return "\n".join(lines)


def select_parameters(args: argparse.Namespace) -> list[Parameter]:
    if args.knobs:
        names = [name.strip() for name in args.knobs.split(",") if name.strip()]
        unknown = [name for name in names if name not in PARAMETER_BY_NAME]
        if unknown:
            raise TunerError(f"knob sconosciuti: {', '.join(unknown)}")
        selected = [PARAMETER_BY_NAME[name] for name in names]
    else:
        selected = [
            parameter for parameter in PARAMETERS if args.profile in parameter.profiles
        ]
    numeric = [parameter.name for parameter in selected if parameter.quality == "numeric"]
    if numeric and not args.allow_numeric:
        if args.knobs:
            raise TunerError(
                f"i knob {', '.join(numeric)} richiedono --allow-numeric"
            )
        selected = [parameter for parameter in selected if parameter.quality != "numeric"]
    dependency_order = {
        "DS4_MULTI_QUANT_CACHE": 0,
        "DS4_EXPERT_CACHE_SLOTS": 1,
        "DS4_EXPERT_CACHE_UNIFORM": 2,
        "DS4_POOL_INTERLEAVE": 3,
        "DS4_EXPERT_LOOKAHEAD": 4,
    }
    original_order = {parameter.name: index for index, parameter in enumerate(PARAMETERS)}
    selected.sort(
        key=lambda parameter: (
            dependency_order.get(parameter.name, 10), original_order[parameter.name]
        )
    )
    return selected


def build_demo(args: argparse.Namespace) -> None:
    if args.skip_build or args.dry_run:
        return
    environment = dict(os.environ)
    xcode = Path("/Applications/Xcode.app/Contents/Developer")
    if "DEVELOPER_DIR" not in environment and xcode.is_dir():
        environment["DEVELOPER_DIR"] = str(xcode)
    print("[build] swift build -c release --product DS4Demo")
    subprocess.run(
        ["swift", "build", "-c", "release", "--product", "DS4Demo"],
        cwd=ROOT, env=environment, check=True,
    )


def active_downloads_near(model: Path, recent_seconds: float = 600.0) -> list[Path]:
    """Return recently modified partial downloads in the model directory.

    A concurrent multi-gigabyte download invalidates SSD tuning.  Old/stale
    ``.part`` files are ignored so an interrupted download does not block the
    tuner forever.
    """
    now = time.time()
    found: list[Path] = []
    for pattern in ("*.part", "*.download", "*.crdownload"):
        for path in model.parent.glob(pattern):
            try:
                if now - path.stat().st_mtime <= recent_seconds:
                    found.append(path)
            except FileNotFoundError:
                continue
    return sorted(set(found))


def self_test() -> int:
    assert ordered_neighbors(("1", "2", "3"), "2") == [("3", 1), ("1", -1)]
    assert ordered_neighbors(("1", "2", "3"), "1") == [("2", 1)]
    assert ordered_neighbors(("1", "2"), "9") == []
    with tempfile.TemporaryDirectory(prefix="ds4-autotune-selftest-") as tmp:
        tmp_path = Path(tmp)
        exact_a = str(tmp_path / "exact-a")
        exact_b = str(tmp_path / "exact-b")
        near = str(tmp_path / "near")
        top_swap = str(tmp_path / "top-swap")
        vectors = [[1.0, 0.5, 0.25, -1.0], [2.0, 1.0, 0.0, -2.0]]
        write_synthetic(exact_a, vectors, [0, 0])
        write_synthetic(exact_b, vectors, [0, 0])
        write_synthetic(near, [[1.0, 0.500001, 0.25, -1.0], vectors[1]], [0, 0])
        write_synthetic(
            top_swap,
            [[1.0, 0.49999, 0.50000, -1.0], vectors[1]],
            [0, 0],
        )
        assert quality_summary(exact_a, exact_b, "exact")["passed"]
        assert not quality_summary(exact_a, exact_b, "typo")["passed"]
        assert not quality_summary(exact_a, near, "exact")["passed"]
        assert quality_summary(exact_a, near, "numeric")["passed"]
        # Same argmax and tiny deltas, but ordered top-3 changed: reject.
        base_swap = str(tmp_path / "base-swap")
        write_synthetic(base_swap, [[1.0, 0.50000, 0.49999, -1.0], vectors[1]], [0, 0])
        swapped = quality_summary(base_swap, top_swap, "numeric")
        assert not swapped["passed"] and not swapped["top3Same"]
        Path(top_swap + ".f32").write_bytes(b"bad")
        assert not quality_summary(base_swap, top_swap, "numeric")["passed"]

        log = tmp_path / "sample.log"
        log.write_text(
            "DS4Demo: prompt 'x' (1 car.) -> 25 tokens; generating 64 (greedy, streaming)…\n"
            "  [tok 1  0.5s  2.00 tok/s]\n"
            "  [tok 2  0.4s  2.50 tok/s]\n"
            "  [tok 3  0.4s  2.50 tok/s]\n"
            "  [tok 4  0.4s  2.50 tok/s]\n"
            "  [tok 5  0.3s  3.00 tok/s]\n"
            "  [tok 6  0.3s  3.00 tok/s]\n"
            "Profilo decode — 2 token, 2 iterazioni-layer\n"
            "  route/attn    100.0 ms/token\n"
            "  gather IO    200.0 ms/token\n"
            "  experts        3.0 ms/token\n"
            "  gather IO     640.0 MB/token — banda effettiva 4.20 GB/s\n",
            encoding="utf-8",
        )
        diag = parse_diagnostics(str(log), 0)
        assert diag["promptTokens"] == 25
        assert diag["gatherGBs"] == 4.2

        # End-to-end coordinate walk with an instantaneous fake DS4Demo.  Its
        # exact logits never change and throughput rises monotonically 1→2→3.
        fake = tmp_path / "fake-ds4demo.py"
        fake.write_text(
            """#!/usr/bin/env python3
import json, os, struct
from pathlib import Path
value = int(os.environ.get("DS4_FAKE_KNOB", "1"))
tps = {1: 1.00, 2: 1.10, 3: 1.20}[value]
trace = os.environ["DS4_AB_TRACE"]
vectors = [[1.0, 0.5, 0.25, -1.0] for _ in range(8)]
raw = bytearray()
frames = []
offset = 0
for index, vector in enumerate(vectors):
    packed = struct.pack("<4f", *vector)
    raw.extend(packed)
    digest = 0xcbf29ce484222325
    for byte in packed:
        digest ^= byte
        digest = (digest * 0x100000001b3) & 0xffffffffffffffff
    frames.append({
        "phase": "prefill" if index == 0 else "decode",
        "step": index,
        "inputToken": 0,
        "offsetFloats": offset,
        "count": 4,
        "finiteCount": 4,
        "bitHashFNV1a64": f"{digest:016x}",
        "top3": [
            {"token": 0, "logit": 1.0},
            {"token": 1, "logit": 0.5},
            {"token": 2, "logit": 0.25},
        ],
    })
    offset += 4
Path(trace + ".f32").write_bytes(raw)
Path(trace + ".json").write_text(json.dumps({
    "format": "ds4-ab-logits-v1",
    "byteOrder": "little",
    "floatFormat": "ieee754-f32",
    "traceLimit": 8,
    "capturedFrames": 8,
    "generatedTokens": [0] * 8,
    "frames": frames,
}))
print("DS4Demo: prompt 'fake' (4 car.) -> 128 tokens; generating 8 (greedy, streaming)…")
print("DS4Demo: prefill 128 token (layer-major) in 12.8s (10.00 tok/s)")
for index in range(1, 9):
    print(f"  [tok {index}  0.1s  {tps:.2f} tok/s]")
print(f"DS4Demo: 8 tokens in {8/tps:.2f}s ({tps:.2f} tok/s)")
print(f"DS4Demo: REGIME (dal token 2): 7 token in {7/tps:.2f}s ({tps:.2f} tok/s)")
print("Profilo decode — 7 token, 7 iterazioni-layer")
print("  route/attn    100.0 ms/token")
print("  gather IO     50.0 ms/token")
print("  experts        3.0 ms/token")
print(f"  totale        {1000/tps:.1f} ms/token  (~{tps:.2f} tok/s)")
""",
            encoding="utf-8",
        )
        fake.chmod(0o755)
        dummy_model = tmp_path / "model.gguf"
        dummy_prompt = tmp_path / "prompt.txt"
        dummy_model.write_bytes(b"fake")
        dummy_prompt.write_text("fake", encoding="utf-8")
        invalid_threshold_args = parser().parse_args([
            str(dummy_model), str(dummy_prompt), "--dry-run", "--numeric-atol", "inf"
        ])
        try:
            validate_args(invalid_threshold_args)
        except TunerError:
            pass
        else:
            raise AssertionError("una soglia numerica infinita deve essere rifiutata")
        output = tmp_path / "autotune"
        fake_args = parser().parse_args([
            str(dummy_model), str(dummy_prompt),
            "--binary", str(fake), "--output", str(output),
            "--preset", "m1pro16-mixed", "--usage-seed", "off",
            "--skip-build", "--max-passes", "1", "--max-new", "8",
            "--warmup", "1", "--trace-frames", "8", "--cooldown", "0",
            "--timeout", "30",
        ])
        fake_parameter = Parameter(
            "DS4_FAKE_KNOB", ("1", "2", "3"), "1",
            profiles=("standard",), description="self-test",
        )
        tuner = Autotuner(fake_args, [fake_parameter])
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                tuner.tune()
            assert tuner.current_config["DS4_FAKE_KNOB"] == "3"
            assert tuner.status == "complete"
            assert sum(trial.selected for trial in tuner.trials) == 2
            assert (output / "report.md").is_file()
            assert "DS4_FAKE_KNOB=3" in (output / "final-env.sh").read_text(encoding="utf-8")
        finally:
            tuner.lock.release()

        # Resume must skip an orphan directory left after a hard crash.
        (output / "runs" / "0099-orphan").mkdir()
        resume_args = parser().parse_args([
            str(dummy_model), str(dummy_prompt),
            "--binary", str(fake), "--output", str(output),
            "--preset", "m1pro16-mixed", "--usage-seed", "off",
            "--skip-build", "--max-passes", "1", "--max-new", "8",
            "--warmup", "1", "--trace-frames", "8", "--cooldown", "0",
            "--timeout", "30", "--resume", "--continue-complete",
        ])
        resumed = Autotuner(resume_args, [fake_parameter])
        try:
            assert resumed.run_counter == 99
            assert resumed.current_config["DS4_FAKE_KNOB"] == "3"
        finally:
            resumed.lock.release()

        # Skipping the final ABBA must never produce a VALIDATED environment.
        unvalidated_output = tmp_path / "autotune-unvalidated"
        unvalidated_args = parser().parse_args([
            str(dummy_model), str(dummy_prompt),
            "--binary", str(fake), "--output", str(unvalidated_output),
            "--preset", "m1pro16-mixed", "--usage-seed", "off",
            "--skip-build", "--max-passes", "1", "--max-new", "8",
            "--warmup", "1", "--trace-frames", "8", "--cooldown", "0",
            "--timeout", "30", "--no-final-validation",
        ])
        fixed_parameter = Parameter(
            "DS4_FAKE_KNOB", ("1",), "1",
            profiles=("standard",), description="self-test no final",
        )
        unvalidated = Autotuner(unvalidated_args, [fixed_parameter])
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                unvalidated.tune()
            final_text = (unvalidated_output / "final-env.sh").read_text(encoding="utf-8")
            assert unvalidated.status == "complete_unvalidated"
            assert "# NOT FINAL:" in final_text and "# VALIDATED:" not in final_text
        finally:
            unvalidated.lock.release()
    print("metal_autotune.py self-test: PASS")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("gguf", nargs="?", help="modello GGUF")
    result.add_argument("prompt", nargs="?", help="file prompt per decode")
    result.add_argument(
        "--prefill-prompt", help="file separato >=1024 token per i knob prefill"
    )
    result.add_argument("--output", help="directory artefatti/checkpoint")
    result.add_argument("--binary", default=str(DEFAULT_BINARY), help="binario DS4Demo release")
    result.add_argument(
        "--profile", choices=("io", "standard", "full", "prefill", "numeric"),
        default="standard",
    )
    result.add_argument("--knobs", help="lista DS4_* separata da virgole (override del profilo)")
    result.add_argument("--preset", choices=("m1pro16-mixed", "inherit"), default="m1pro16-mixed")
    result.add_argument("--usage-seed", default="auto", help="auto, off o file usage JSON")
    result.add_argument(
        "--allow-cold-usage", action="store_true",
        help="tuna lookahead/allocazione senza usage seed",
    )
    result.add_argument(
        "--allow-numeric", action="store_true",
        help="consenti knob non bit-exact con gate numerico stretto",
    )
    result.add_argument("--max-new", type=int, default=64)
    result.add_argument(
        "--context", type=int,
        help="DS4_DEMO_CONTEXT della configurazione reale da ottimizzare",
    )
    result.add_argument("--warmup", type=int, default=4)
    result.add_argument("--trace-frames", type=int, default=64)
    result.add_argument("--max-passes", type=int, default=2)
    result.add_argument("--min-gain", type=float, default=0.02)
    result.add_argument("--screen-regression", type=float, default=0.03)
    result.add_argument("--max-pair-regression", type=float, default=0.03)
    result.add_argument("--max-secondary-regression", type=float, default=0.08)
    result.add_argument("--stability-floor", type=float, default=0.75)
    result.add_argument("--baseline-retries", type=int, default=1)
    result.add_argument("--numeric-atol", type=float, default=1e-4)
    result.add_argument("--numeric-rtol", type=float, default=1e-5)
    result.add_argument("--numeric-max-abs", type=float, default=1e-3)
    result.add_argument("--numeric-nrmse", type=float, default=1e-5)
    result.add_argument(
        "--min-memory-free-percent", type=float, default=8.0,
        help="respinge run che terminano sotto questa memoria libera (memory_pressure -Q)",
    )
    result.add_argument(
        "--max-swapout-mib", type=float, default=128.0,
        help="respinge run che causano più swapout di questa soglia",
    )
    result.add_argument("--cooldown", type=float, default=2.0)
    result.add_argument("--timeout", type=float, default=1800.0)
    result.add_argument("--skip-build", action="store_true")
    result.add_argument("--keep-traces", action="store_true")
    result.add_argument(
        "--allow-active-download", action="store_true",
        help="esegui anche se un file .part recente è nella directory del modello",
    )
    result.add_argument("--resume", action="store_true")
    result.add_argument("--continue-complete", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--no-final-validation", dest="final_validation", action="store_false")
    result.set_defaults(final_validation=True)
    result.add_argument("--self-test", action="store_true")
    return result


def validate_args(args: argparse.Namespace) -> None:
    if args.self_test:
        return
    if not args.gguf or not args.prompt:
        raise TunerError("gguf e prompt sono obbligatori (oppure usa --self-test)")
    paths = (
        (Path(args.gguf).expanduser(), "GGUF"),
        (Path(args.prompt).expanduser(), "prompt"),
    )
    for path, label in paths:
        if not path.is_file():
            raise TunerError(f"{label} non trovato: {path}")
    if args.prefill_prompt and not Path(args.prefill_prompt).expanduser().is_file():
        raise TunerError(f"prefill prompt non trovato: {args.prefill_prompt}")
    if args.resume and not args.output:
        raise TunerError("--resume richiede la stessa --output del run interrotto")
    if args.continue_complete and not args.resume:
        raise TunerError("--continue-complete richiede --resume")
    if not args.dry_run and not args.allow_active_download:
        active = active_downloads_near(Path(args.gguf).expanduser().resolve())
        if active:
            rendered = ", ".join(str(path) for path in active)
            raise TunerError(
                "download recente nella directory del modello: i risultati SSD sarebbero falsati. "
                f"Attendi che finisca oppure usa consapevolmente --allow-active-download ({rendered})"
            )
    if args.max_new < 8:
        raise TunerError("--max-new deve essere >=8")
    if args.context is not None and args.context < 128:
        raise TunerError("--context deve essere >=128")
    if not 0 <= args.warmup < args.max_new:
        raise TunerError("--warmup deve essere >=0 e < --max-new")
    if not 2 <= args.trace_frames <= 64:
        raise TunerError("--trace-frames deve essere 2...64")
    minimum_trace = min(args.max_new, 64)
    if args.trace_frames < minimum_trace:
        raise TunerError(
            f"--trace-frames deve essere >=min(max-new,64)={minimum_trace} "
            "per coprire ogni decisione generata"
        )
    if args.max_passes < 1:
        raise TunerError("--max-passes deve essere >=1")
    if args.baseline_retries < 0:
        raise TunerError("--baseline-retries deve essere >=0")
    bounded = (
        "min_gain", "screen_regression", "max_pair_regression",
        "max_secondary_regression", "stability_floor",
    )
    for name in bounded:
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise TunerError(f"--{name.replace('_', '-')} deve essere finito e in [0,1]")
    nonnegative = (
        "numeric_atol", "numeric_rtol", "numeric_max_abs", "numeric_nrmse",
        "cooldown", "max_swapout_mib",
    )
    for name in nonnegative:
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise TunerError(f"--{name.replace('_', '-')} deve essere finito e >=0")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise TunerError("--timeout deve essere finito e >0")
    if (
        not math.isfinite(args.min_memory_free_percent)
        or not 0 <= args.min_memory_free_percent <= 100
    ):
        raise TunerError("--min-memory-free-percent deve essere finito e in [0,100]")


def dry_run(args: argparse.Namespace, parameters: list[Parameter]) -> int:
    inherited = {
        key: value for key, value in os.environ.items()
        if key.startswith("DS4_") and key not in TRANSIENT_DS4_KEYS
        and not key.startswith("DS4_AB_")
    }
    if args.preset == "inherit":
        missing = [parameter.name for parameter in parameters if parameter.name not in inherited]
        if missing:
            raise TunerError(
                "--preset inherit richiede valori espliciti; mancano: " + ", ".join(missing)
            )
    environment = {} if args.preset == "inherit" else dict(M1PRO16_MIXED_PRESET)
    environment.update(inherited)
    if args.context is not None:
        environment["DS4_DEMO_CONTEXT"] = str(args.context)
    print(f"Profilo: {args.profile}; parametri: {len(parameters)}")
    print("La ricerca usa 2 run per screening e 4 per ogni candidato non regressivo/promettente.")
    for parameter in parameters:
        current = environment.get(parameter.name, parameter.default)
        print(
            f"- {parameter.name}: corrente={current}, valori={','.join(parameter.values)}, "
            f"metrica={parameter.metric}, qualità={parameter.quality}, "
            f"RAM-sensitive={'sì' if parameter.memory_risk else 'no'}"
        )
    return 0


def main() -> int:
    args = parser().parse_args()
    try:
        validate_args(args)
        if args.self_test:
            return self_test()
        parameters = select_parameters(args)
        if not parameters:
            raise TunerError("nessun parametro selezionato")
        if args.dry_run:
            return dry_run(args, parameters)
        build_demo(args)
        if not Path(args.binary).expanduser().is_file():
            raise TunerError(f"DS4Demo non trovato: {args.binary}")
        tuner = Autotuner(args, parameters)
        try:
            tuner.tune()
        except KeyboardInterrupt:
            tuner.status = "interrupted"
            tuner._checkpoint()
            print(f"\nInterrotto. Riprendi con --resume --output {shlex.quote(str(tuner.output))}", file=sys.stderr)
            return 130
        except Exception:
            if tuner.status != "validation_failed":
                tuner.status = "failed"
            tuner._checkpoint()
            raise
        finally:
            tuner.lock.release()
        return 0
    except TunerError as error:
        print(f"metal_autotune: ERRORE: {error}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as error:
        print(f"metal_autotune: build fallita (exit {error.returncode})", file=sys.stderr)
        return error.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
