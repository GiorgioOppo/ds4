#!/usr/bin/env python3
"""Compare bounded DS4Demo A/B logit traces and timing logs.

The implementation intentionally uses only the Python standard library.  Raw
Float32 vectors are memory-mapped and compared one element at a time, so the
analyzer's RAM does not grow with vocab_size * frame_count.
"""

from __future__ import annotations

import argparse
import json
import math
import mmap
import os
import re
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


FORMAT = "ds4-ab-logits-v1"
FLOAT_FORMAT = "ieee754-f32"
MAX_TRACE_FRAMES = 64
# Current DS4/Qwen vocabularies are far below this ceiling.  Keeping a hard
# upper bound prevents a corrupt sparse raw file from making the validator
# spend hours walking billions of declared floats.
MAX_VECTOR_FLOATS = 1_000_000
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
FNV_MASK = 0xFFFFFFFFFFFFFFFF


@dataclass
class VectorStats:
    count: int = 0
    exact: int = 0
    within: int = 0
    nonfinite: int = 0
    max_abs: float = 0.0
    max_rel: float = 0.0
    squared_error: float = 0.0
    squared_baseline: float = 0.0

    def merge(self, other: "VectorStats") -> None:
        self.count += other.count
        self.exact += other.exact
        self.within += other.within
        self.nonfinite += other.nonfinite
        self.max_abs = max(self.max_abs, other.max_abs)
        self.max_rel = max(self.max_rel, other.max_rel)
        self.squared_error += other.squared_error
        self.squared_baseline += other.squared_baseline

    @property
    def nrmse(self) -> float:
        if self.squared_error == 0:
            return 0.0
        return math.sqrt(self.squared_error / max(self.squared_baseline, 1e-30))


@dataclass(frozen=True)
class FrameLayout:
    phase: str
    step: int
    input_token: int | None
    offset_floats: int
    count: int
    finite_count: int
    bit_hash: str
    top3: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class TraceLayout:
    prefix: str
    trace_limit: int
    generated_tokens: tuple[int, ...]
    frames: tuple[FrameLayout, ...]
    total_floats: int


@dataclass(frozen=True)
class RawFrameSummary:
    finite_count: int
    bit_hash: str
    top3: tuple[tuple[int, float], ...]

    @property
    def token_ids(self) -> list[int]:
        return [token for token, _ in self.top3]


def load_document(prefix: str) -> dict[str, Any]:
    path = Path(prefix + ".json")
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"documento trace non-oggetto in {path}")
    if document.get("format") != FORMAT:
        raise ValueError(f"formato trace non supportato in {path}: {document.get('format')!r}")
    if document.get("byteOrder") != "little" or sys.byteorder != "little":
        raise ValueError("il comparatore richiede trace e host little-endian")
    return document


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} deve essere un intero")
    return value


def validate_document(prefix: str, document: dict[str, Any]) -> TraceLayout:
    """Validate all bounds and cross-field relationships before mmap access."""

    if document.get("floatFormat") != FLOAT_FORMAT:
        raise ValueError(
            f"{prefix}: floatFormat non supportato: {document.get('floatFormat')!r}"
        )
    trace_limit = require_int(document.get("traceLimit"), f"{prefix}: traceLimit")
    if not 1 <= trace_limit <= MAX_TRACE_FRAMES:
        raise ValueError(
            f"{prefix}: traceLimit {trace_limit} fuori da 1...{MAX_TRACE_FRAMES}"
        )
    captured = require_int(
        document.get("capturedFrames"), f"{prefix}: capturedFrames"
    )
    if not 1 <= captured <= trace_limit:
        raise ValueError(
            f"{prefix}: capturedFrames {captured} fuori da 1...traceLimit({trace_limit})"
        )

    raw_frames = document.get("frames")
    if not isinstance(raw_frames, list):
        raise ValueError(f"{prefix}: frames deve essere un array")
    if len(raw_frames) != captured:
        raise ValueError(
            f"{prefix}: capturedFrames={captured}, ma frames contiene {len(raw_frames)} elementi"
        )
    raw_tokens = document.get("generatedTokens")
    if not isinstance(raw_tokens, list):
        raise ValueError(f"{prefix}: generatedTokens deve essere un array")
    generated_tokens = tuple(
        require_int(token, f"{prefix}: generatedTokens[{index}]")
        for index, token in enumerate(raw_tokens)
    )
    # The trace captures prefill plus one frame after each fed generated token,
    # stopping at traceLimit.  This catches missing/stale frame arrays while
    # still allowing EOS to end generation before the requested maximum.
    expected_captured = min(trace_limit, len(generated_tokens) + 1)
    if captured != expected_captured:
        raise ValueError(
            f"{prefix}: capturedFrames={captured}, atteso {expected_captured} "
            "da traceLimit e generatedTokens"
        )

    layouts: list[FrameLayout] = []
    expected_offset = 0
    vocab_count: int | None = None
    hash_pattern = re.compile(r"^[0-9a-f]{16}$")
    for index, raw_frame in enumerate(raw_frames):
        label = f"{prefix}: frame {index}"
        if not isinstance(raw_frame, dict):
            raise ValueError(f"{label} deve essere un oggetto")
        phase = raw_frame.get("phase")
        expected_phase = "prefill" if index == 0 else "decode"
        if phase != expected_phase:
            raise ValueError(f"{label}: phase {phase!r}, attesa {expected_phase!r}")
        step = require_int(raw_frame.get("step"), f"{label}: step")
        if step != index:
            raise ValueError(f"{label}: step {step}, atteso {index}")

        input_value = raw_frame.get("inputToken")
        if input_value is None:
            input_token = None
        else:
            input_token = require_int(input_value, f"{label}: inputToken")
        if index > 0:
            expected_input = generated_tokens[index - 1]
            if input_token != expected_input:
                raise ValueError(
                    f"{label}: inputToken {input_token!r}, atteso generatedTokens"
                    f"[{index - 1}]={expected_input}"
                )

        offset = require_int(raw_frame.get("offsetFloats"), f"{label}: offsetFloats")
        if offset != expected_offset:
            raise ValueError(
                f"{label}: offsetFloats {offset}, atteso offset contiguo {expected_offset}"
            )
        count = require_int(raw_frame.get("count"), f"{label}: count")
        if not 3 <= count <= MAX_VECTOR_FLOATS:
            raise ValueError(
                f"{label}: count {count} fuori da 3...{MAX_VECTOR_FLOATS}"
            )
        if vocab_count is None:
            vocab_count = count
        elif count != vocab_count:
            raise ValueError(
                f"{label}: count {count}, atteso vocab uniforme {vocab_count}"
            )

        finite_count = require_int(
            raw_frame.get("finiteCount"), f"{label}: finiteCount"
        )
        if not 0 <= finite_count <= count:
            raise ValueError(
                f"{label}: finiteCount {finite_count} fuori da 0...{count}"
            )
        bit_hash = raw_frame.get("bitHashFNV1a64")
        if not isinstance(bit_hash, str) or hash_pattern.fullmatch(bit_hash) is None:
            raise ValueError(f"{label}: bitHashFNV1a64 non valido")

        raw_top = raw_frame.get("top3")
        if not isinstance(raw_top, list) or len(raw_top) != 3:
            raise ValueError(f"{label}: top3 deve contenere esattamente 3 elementi")
        top: list[tuple[int, float]] = []
        seen_tokens: set[int] = set()
        for rank, item in enumerate(raw_top):
            if not isinstance(item, dict):
                raise ValueError(f"{label}: top3[{rank}] deve essere un oggetto")
            token = require_int(item.get("token"), f"{label}: top3[{rank}].token")
            if not 0 <= token < count or token in seen_tokens:
                raise ValueError(f"{label}: token top3 non valido/duplicato: {token}")
            logit_value = item.get("logit")
            if isinstance(logit_value, bool) or not isinstance(logit_value, (int, float)):
                raise ValueError(f"{label}: top3[{rank}].logit non numerico")
            logit = float(logit_value)
            if not math.isfinite(logit):
                raise ValueError(f"{label}: top3[{rank}].logit non finito")
            try:
                # JSON carries the shortest decimal that round-trips to Swift
                # Float.  Normalize the Python double back to Float32 before
                # comparing it with the raw mmap value.
                logit = struct.unpack("<f", struct.pack("<f", logit))[0]
            except (OverflowError, struct.error) as error:
                raise ValueError(
                    f"{label}: top3[{rank}].logit fuori dal range Float32"
                ) from error
            if not math.isfinite(logit):
                raise ValueError(f"{label}: top3[{rank}].logit Float32 non finito")
            seen_tokens.add(token)
            top.append((token, logit))

        expected_offset += count
        layouts.append(
            FrameLayout(
                phase=phase,
                step=step,
                input_token=input_token,
                offset_floats=offset,
                count=count,
                finite_count=finite_count,
                bit_hash=bit_hash,
                top3=tuple(top),
            )
        )

    assert vocab_count is not None
    for index, token in enumerate(generated_tokens):
        if not 0 <= token < vocab_count:
            raise ValueError(
                f"{prefix}: generatedTokens[{index}]={token} fuori dal vocab {vocab_count}"
            )
    first_input = layouts[0].input_token
    if first_input is not None and not 0 <= first_input < vocab_count:
        raise ValueError(f"{prefix}: inputToken prefill {first_input} fuori dal vocab")

    raw_path = Path(prefix + ".f32")
    expected_bytes = expected_offset * 4
    actual_bytes = raw_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{prefix}: dimensione raw {actual_bytes}, attesa {expected_bytes}"
        )
    return TraceLayout(
        prefix=prefix,
        trace_limit=trace_limit,
        generated_tokens=generated_tokens,
        frames=tuple(layouts),
        total_floats=expected_offset,
    )


def compare_vector(
    baseline_map: mmap.mmap,
    candidate_map: mmap.mmap,
    baseline_offset: int,
    candidate_offset: int,
    count: int,
    atol: float,
    rtol: float,
) -> tuple[VectorStats, RawFrameSummary, RawFrameSummary]:
    b_start = baseline_offset * 4
    c_start = candidate_offset * 4
    byte_count = count * 4
    b_bytes = memoryview(baseline_map)[b_start : b_start + byte_count]
    c_bytes = memoryview(candidate_map)[c_start : c_start + byte_count]
    b_float = b_bytes.cast("f")
    c_float = c_bytes.cast("f")
    b_bits = b_bytes.cast("I")
    c_bits = c_bytes.cast("I")
    stats = VectorStats(count=count)
    b_finite = 0
    c_finite = 0
    b_hash = FNV_OFFSET
    c_hash = FNV_OFFSET
    b_top: list[tuple[int, float]] = []
    c_top: list[tuple[int, float]] = []

    def update_top(top: list[tuple[int, float]], token: int, value: float) -> None:
        insertion = len(top)
        for rank, (old_token, old_value) in enumerate(top):
            if value > old_value or (value == old_value and token < old_token):
                insertion = rank
                break
        if insertion < 3:
            top.insert(insertion, (token, value))
            if len(top) > 3:
                top.pop()

    def update_hash(value: int, current: int) -> int:
        for shift in (0, 8, 16, 24):
            current ^= (value >> shift) & 0xFF
            current = (current * FNV_PRIME) & FNV_MASK
        return current

    try:
        for index in range(count):
            b = float(b_float[index])
            c = float(c_float[index])
            b_word = int(b_bits[index])
            c_word = int(c_bits[index])
            b_hash = update_hash(b_word, b_hash)
            c_hash = update_hash(c_word, c_hash)
            if math.isfinite(b):
                b_finite += 1
                update_top(b_top, index, b)
            if math.isfinite(c):
                c_finite += 1
                update_top(c_top, index, c)
            if b_word == c_word:
                stats.exact += 1
            if not math.isfinite(b) or not math.isfinite(c):
                stats.nonfinite += 1
                continue
            delta = abs(c - b)
            stats.max_abs = max(stats.max_abs, delta)
            stats.max_rel = max(stats.max_rel, delta / max(abs(b), 1e-12))
            stats.squared_error += delta * delta
            stats.squared_baseline += b * b
            if delta <= atol + rtol * abs(b):
                stats.within += 1
    finally:
        # mmap.close() raises BufferError while any exported memoryview lives.
        b_bits.release()
        c_bits.release()
        b_float.release()
        c_float.release()
        b_bytes.release()
        c_bytes.release()
    return (
        stats,
        RawFrameSummary(
            finite_count=b_finite,
            bit_hash=f"{b_hash:016x}",
            top3=tuple(b_top),
        ),
        RawFrameSummary(
            finite_count=c_finite,
            bit_hash=f"{c_hash:016x}",
            top3=tuple(c_top),
        ),
    )


def validate_raw_summary(
    layout: FrameLayout,
    summary: RawFrameSummary,
    trace_label: str,
    frame_index: int,
    structural: list[str],
) -> None:
    label = f"{trace_label} frame {frame_index}"
    if layout.finite_count != summary.finite_count:
        structural.append(
            f"{label}: finiteCount metadata {layout.finite_count}, raw {summary.finite_count}"
        )
    if layout.bit_hash != summary.bit_hash:
        structural.append(
            f"{label}: hash metadata {layout.bit_hash}, raw {summary.bit_hash}"
        )
    if layout.top3 != summary.top3:
        structural.append(
            f"{label}: top3 metadata {layout.top3!r}, raw {summary.top3!r}"
        )


def compare_traces(
    baseline_prefix: str, candidate_prefix: str, atol: float, rtol: float
) -> dict[str, Any]:
    baseline_document = load_document(baseline_prefix)
    candidate_document = load_document(candidate_prefix)
    baseline = validate_document(baseline_prefix, baseline_document)
    candidate = validate_document(candidate_prefix, candidate_document)
    structural: list[str] = []
    if baseline.trace_limit != candidate.trace_limit:
        structural.append(
            f"traceLimit diverso: {baseline.trace_limit} vs {candidate.trace_limit}"
        )
    b_frames = baseline.frames
    c_frames = candidate.frames
    if len(b_frames) != len(c_frames):
        structural.append(f"numero frame diverso: {len(b_frames)} vs {len(c_frames)}")
    if not any(frame.phase == "decode" for frame in b_frames):
        structural.append("trace baseline senza frame decode (EOS immediato o trace troppo corta)")
    if not any(frame.phase == "decode" for frame in c_frames):
        structural.append("trace candidate senza frame decode (EOS immediato o trace troppo corta)")

    raw_b_path = Path(baseline_prefix + ".f32")
    raw_c_path = Path(candidate_prefix + ".f32")

    aggregate = VectorStats()
    rows: list[dict[str, Any]] = []
    argmax_same = True
    top3_same = True
    with raw_b_path.open("rb") as b_handle, raw_c_path.open("rb") as c_handle:
        b_map = mmap.mmap(b_handle.fileno(), 0, access=mmap.ACCESS_READ)
        c_map = mmap.mmap(c_handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for index, (b_frame, c_frame) in enumerate(zip(b_frames, c_frames)):
                b_key = (b_frame.phase, b_frame.step)
                c_key = (c_frame.phase, c_frame.step)
                if b_key != c_key:
                    structural.append(f"frame {index}: chiave {b_key} vs {c_key}")
                if b_frame.input_token != c_frame.input_token:
                    structural.append(
                        f"frame {index}: inputToken {b_frame.input_token} vs "
                        f"{c_frame.input_token}"
                    )
                b_count = b_frame.count
                c_count = c_frame.count
                if b_count != c_count:
                    structural.append(f"frame {index}: vocab {b_count} vs {c_count}")
                    continue
                stats, b_summary, c_summary = compare_vector(
                    b_map,
                    c_map,
                    b_frame.offset_floats,
                    c_frame.offset_floats,
                    b_count,
                    atol,
                    rtol,
                )
                aggregate.merge(stats)
                validate_raw_summary(b_frame, b_summary, "baseline", index, structural)
                validate_raw_summary(c_frame, c_summary, "candidate", index, structural)
                b_top = b_summary.token_ids
                c_top = c_summary.token_ids
                frame_argmax_same = bool(b_top and c_top and b_top[0] == c_top[0])
                frame_top3_same = b_top == c_top
                argmax_same = argmax_same and frame_argmax_same
                top3_same = top3_same and frame_top3_same
                if index < len(baseline.generated_tokens) and (
                    not b_top or b_top[0] != baseline.generated_tokens[index]
                ):
                    structural.append(
                        f"baseline frame {index}: argmax raw {b_top[0] if b_top else None}, "
                        f"generatedTokens[{index}]={baseline.generated_tokens[index]}"
                    )
                if index < len(candidate.generated_tokens) and (
                    not c_top or c_top[0] != candidate.generated_tokens[index]
                ):
                    structural.append(
                        f"candidate frame {index}: argmax raw {c_top[0] if c_top else None}, "
                        f"generatedTokens[{index}]={candidate.generated_tokens[index]}"
                    )
                rows.append(
                    {
                        "key": f"{b_key[0]}:{b_key[1]}",
                        "argmaxBaseline": b_top[0] if b_top else None,
                        "argmaxCandidate": c_top[0] if c_top else None,
                        "top3Same": frame_top3_same,
                        "exactPercent": 100.0 * stats.exact / max(stats.count, 1),
                        "withinPercent": 100.0 * stats.within / max(stats.count, 1),
                        "maxAbs": stats.max_abs,
                        "nrmse": stats.nrmse,
                    }
                )
        finally:
            b_map.close()
            c_map.close()

    generated_same = baseline.generated_tokens == candidate.generated_tokens
    exact = (
        not structural
        and generated_same
        and argmax_same
        and top3_same
        and aggregate.nonfinite == 0
        and aggregate.count > 0
        and aggregate.exact == aggregate.count
    )
    numeric = (
        not structural
        and generated_same
        and argmax_same
        and top3_same
        and aggregate.nonfinite == 0
        and aggregate.count > 0
        and aggregate.within == aggregate.count
    )
    verdict = "PASS_EXACT" if exact else "PASS_NUMERIC" if numeric else "FAIL"
    return {
        "verdict": verdict,
        "structuralErrors": structural,
        "generatedSame": generated_same,
        "baselineTokens": list(baseline.generated_tokens),
        "candidateTokens": list(candidate.generated_tokens),
        "argmaxSame": argmax_same,
        "top3Same": top3_same,
        "frameRows": rows,
        "stats": aggregate,
    }


def parse_performance(path: str) -> dict[str, float | None]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")

    def last(pattern: str, flags: int = 0, group: int = 1) -> float | None:
        found = list(re.finditer(pattern, text, flags))
        return float(found[-1].group(group)) if found else None

    return {
        "prefillTps": last(
            r"DS4Demo: prefill \d+ token .*?\(([0-9]+(?:\.[0-9]+)?) tok/s\)"
        ),
        "decodeProfileTps": last(
            r"Profilo decode[\s\S]*?totale\s+[0-9]+(?:\.[0-9]+)? ms/token\s+\(~([0-9]+(?:\.[0-9]+)?) tok/s\)"
        ),
        "decodeSteadyTps": last(
            r"DS4Demo: REGIME .*?\(([0-9]+(?:\.[0-9]+)?) tok/s\)"
        ),
        "decodeWallTps": last(
            r"DS4Demo: \d+ tokens in [0-9]+(?:\.[0-9]+)?s \(([0-9]+(?:\.[0-9]+)?) tok/s\)"
        ),
    }


def fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def speedup(baseline: float | None, candidate: float | None) -> str:
    if baseline is None or candidate is None or baseline == 0:
        return "—"
    ratio = candidate / baseline
    return f"{ratio:.3f}× ({(ratio - 1) * 100:+.1f}%)"


def markdown_report(
    label: str,
    comparison: dict[str, Any],
    baseline_perf: dict[str, float | None],
    candidate_perf: dict[str, float | None],
    atol: float,
    rtol: float,
) -> str:
    stats: VectorStats = comparison["stats"]
    lines = [
        "# DS4 Metal A/B report",
        "",
        f"Confronto: `{label}`",
        "",
        "## Correttezza",
        "",
        f"Verdetto: **{comparison['verdict']}**",
        "",
        f"- token generati identici: `{comparison['generatedSame']}`",
        f"- argmax di ogni frame identico: `{comparison['argmaxSame']}`",
        f"- top-3 di ogni frame identica: `{comparison['top3Same']}`",
        f"- logits bit-identici: `{stats.exact}/{stats.count}` "
        f"({100.0 * stats.exact / max(stats.count, 1):.6f}%)",
        f"- logits entro `atol={atol:g} + rtol={rtol:g}*abs(base)`: "
        f"`{stats.within}/{stats.count}` "
        f"({100.0 * stats.within / max(stats.count, 1):.6f}%)",
        f"- max absolute error: `{stats.max_abs:.9g}`",
        f"- max relative error: `{stats.max_rel:.9g}`",
        f"- normalized RMSE: `{stats.nrmse:.9g}`",
        f"- valori non finiti: `{stats.nonfinite}`",
    ]
    if comparison["structuralErrors"]:
        lines.extend(["", "Errori strutturali:"])
        lines.extend(f"- {error}" for error in comparison["structuralErrors"])

    lines.extend(
        [
            "",
            "| Frame | Argmax base | Argmax candidato | Top-3 | Bit-exact | Entro tolleranza | Max abs | NRMSE |",
            "|---|---:|---:|:---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison["frameRows"]:
        lines.append(
            f"| {row['key']} | {row['argmaxBaseline']} | {row['argmaxCandidate']} | "
            f"{'sì' if row['top3Same'] else 'no'} | {row['exactPercent']:.6f}% | "
            f"{row['withinPercent']:.6f}% | {row['maxAbs']:.6g} | {row['nrmse']:.6g} |"
        )

    metrics = [
        ("Prefill wall tok/s", "prefillTps"),
        ("Decode profile tok/s", "decodeProfileTps"),
        ("Decode regime tok/s", "decodeSteadyTps"),
        ("Decode wall tok/s", "decodeWallTps"),
    ]
    lines.extend(
        [
            "",
            "## Prestazioni",
            "",
            "| Metrica | Baseline | Candidato | Speedup |",
            "|---|---:|---:|---:|",
        ]
    )
    for title, key in metrics:
        b = baseline_perf[key]
        c = candidate_perf[key]
        lines.append(f"| {title} | {fmt(b)} | {fmt(c)} | {speedup(b, c)} |")

    lines.extend(
        [
            "",
            "## Lettura del verdetto",
            "",
            "- `PASS_EXACT`: token, argmax e ogni bit Float32 coincidono.",
            "- `PASS_NUMERIC`: output, argmax e top-3 coincidono e ogni logit rispetta la tolleranza, ma non tutti i bit coincidono.",
            "- `FAIL`: divergenza di output/argmax/top-3, metadata incoerenti, valore non finito, struttura diversa o logit oltre tolleranza.",
            "",
            "Una sola coppia resta una misura esplorativa: invertire l'ordine dei run o ripetere il test prima di promuovere un knob sui default.",
        ]
    )
    return "\n".join(lines) + "\n"


def fnv1a(values: Iterable[float]) -> str:
    value = FNV_OFFSET
    for item in values:
        for byte in struct.pack("<f", item):
            value ^= byte
            value = (value * FNV_PRIME) & FNV_MASK
    return f"{value:016x}"


def write_synthetic(prefix: str, vectors: list[list[float]], tokens: list[int]) -> None:
    raw = bytearray()
    frames = []
    offset = 0
    for index, vector in enumerate(vectors):
        raw.extend(struct.pack(f"<{len(vector)}f", *vector))
        ranked = sorted(
            (
                (token, struct.unpack("<f", struct.pack("<f", value))[0])
                for token, value in enumerate(vector)
                if math.isfinite(value)
            ),
            key=lambda item: (-item[1], item[0]),
        )[:3]
        frames.append(
            {
                "phase": "prefill" if index == 0 else "decode",
                "step": index,
                "inputToken": None if index == 0 else tokens[index - 1],
                "offsetFloats": offset,
                "count": len(vector),
                "finiteCount": sum(math.isfinite(value) for value in vector),
                "bitHashFNV1a64": fnv1a(vector),
                "top3": [{"token": token, "logit": logit} for token, logit in ranked],
            }
        )
        offset += len(vector)
    Path(prefix + ".f32").write_bytes(raw)
    Path(prefix + ".json").write_text(
        json.dumps(
            {
                "format": FORMAT,
                "byteOrder": "little",
                "floatFormat": "ieee754-f32",
                "traceLimit": len(vectors),
                "capturedFrames": len(vectors),
                "generatedTokens": tokens,
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )


def self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="ds4-metal-ab-") as tmp:
        base = os.path.join(tmp, "base")
        exact = os.path.join(tmp, "exact")
        near = os.path.join(tmp, "near")
        bad = os.path.join(tmp, "bad")
        prefill_only = os.path.join(tmp, "prefill-only")
        stale = os.path.join(tmp, "stale")
        bad_offset = os.path.join(tmp, "bad-offset")
        bad_hash = os.path.join(tmp, "bad-hash")
        nonfinite = os.path.join(tmp, "nonfinite")

        def rejected(left: str, right: str, atol: float = 0.0) -> bool:
            try:
                return compare_traces(left, right, atol, 0.0)["verdict"] == "FAIL"
            except (OSError, ValueError, KeyError, TypeError, BufferError, IndexError):
                return True

        vectors = [[1.0, -2.0, 3.0, 0.5], [0.1, 0.2, 0.3, 0.4]]
        write_synthetic(base, vectors, [2, 3])
        write_synthetic(exact, vectors, [2, 3])
        write_synthetic(near, [[1.0, -2.0, 3.0, 0.500001], vectors[1]], [2, 3])
        write_synthetic(bad, [[4.0, -2.0, 3.0, 0.5], vectors[1]], [0, 3])
        write_synthetic(prefill_only, [vectors[0]], [])
        assert compare_traces(base, exact, 0.0, 0.0)["verdict"] == "PASS_EXACT"
        assert compare_traces(base, near, 1e-5, 0.0)["verdict"] == "PASS_NUMERIC"
        assert compare_traces(base, bad, 1e-5, 0.0)["verdict"] == "FAIL"
        assert compare_traces(prefill_only, prefill_only, 0.0, 0.0)["verdict"] == "FAIL"

        # JSON stale: raw argmax changes from token 0 to token 1 within the
        # numerical tolerance.  Keep the old JSON top3 but update its hash so
        # this specifically exercises raw top3 recomputation, not only hashing.
        stale_base = [[1.0, 0.99995, 0.5, 0.25], [2.0, 1.0, 0.0, -1.0]]
        stale_first = [0.99996, 0.99999, 0.5, 0.25]
        write_synthetic(stale, stale_base, [0, 0])
        stale_raw = bytearray(Path(stale + ".f32").read_bytes())
        stale_raw[:16] = struct.pack("<4f", *stale_first)
        Path(stale + ".f32").write_bytes(stale_raw)
        stale_document = load_document(stale)
        stale_document["frames"][0]["bitHashFNV1a64"] = fnv1a(stale_first)
        Path(stale + ".json").write_text(json.dumps(stale_document), encoding="utf-8")
        stale_reference = os.path.join(tmp, "stale-reference")
        write_synthetic(stale_reference, stale_base, [0, 0])
        assert rejected(stale_reference, stale, 1e-4)

        write_synthetic(bad_offset, vectors, [2, 3])
        offset_document = load_document(bad_offset)
        offset_document["frames"][1]["offsetFloats"] = 0
        Path(bad_offset + ".json").write_text(
            json.dumps(offset_document), encoding="utf-8"
        )
        assert rejected(base, bad_offset)

        write_synthetic(bad_hash, vectors, [2, 3])
        hash_document = load_document(bad_hash)
        hash_document["frames"][0]["bitHashFNV1a64"] = "0000000000000000"
        Path(bad_hash + ".json").write_text(
            json.dumps(hash_document), encoding="utf-8"
        )
        assert rejected(base, bad_hash)

        nonfinite_vectors = [
            [3.0, 2.0, float("inf"), 0.0],
            [4.0, 3.0, 2.0, float("nan")],
        ]
        write_synthetic(nonfinite, nonfinite_vectors, [0, 0])
        assert rejected(nonfinite, nonfinite)
    print("metal_ab_compare.py self-test: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", help="trace prefix della baseline")
    parser.add_argument("--candidate", help="trace prefix del candidato")
    parser.add_argument("--baseline-log", help="log DS4Demo baseline")
    parser.add_argument("--candidate-log", help="log DS4Demo candidato")
    parser.add_argument("--label", default="baseline -> candidate")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    required = [args.baseline, args.candidate, args.baseline_log, args.candidate_log]
    if any(value is None for value in required):
        parser.error("--baseline, --candidate, --baseline-log e --candidate-log sono obbligatori")
    if (
        not math.isfinite(args.atol)
        or not math.isfinite(args.rtol)
        or args.atol < 0
        or args.rtol < 0
    ):
        parser.error("le tolleranze devono essere finite e >= 0")

    try:
        comparison = compare_traces(args.baseline, args.candidate, args.atol, args.rtol)
        report = markdown_report(
            args.label,
            comparison,
            parse_performance(args.baseline_log),
            parse_performance(args.candidate_log),
            args.atol,
            args.rtol,
        )
    except Exception as error:
        print(
            f"metal_ab_compare: trace non valida: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2
    sys.stdout.write(report)
    return 0 if comparison["verdict"] != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
