#!/usr/bin/env python3
"""Overlay two ds4-bench CSV files in one SVG throughput graph.

Same rendering style as plot_speed.py, but with both runs on shared axes so
an A/B comparison (say, SSD streaming with and without the expert-bundle
sidecar) reads at a glance: the baseline run is drawn dashed, the candidate
run solid, and the subtitle reports the candidate's average throughput delta
over the context points the two runs have in common.
"""

import argparse
import html
from pathlib import Path

from plot_speed import (
    AXIS_COLOR,
    GEN_COLOR,
    GRID_COLOR,
    PREFILL_COLOR,
    fmt_tick,
    frange,
    nice_ceil,
    nice_step,
    points_to_polyline,
    read_points,
)

BASE_PREFILL_COLOR = "#93c5fd"
BASE_GEN_COLOR = "#fca5a5"
DASH = "7 5"


def derive_label(csv_path):
    return csv_path.stem.replace("_", " ").replace("-", " ")


def mean_delta_pct(base_rows, cand_rows, index):
    """Average percent change candidate vs baseline over shared ctx points."""
    base = {row[0]: row[index] for row in base_rows}
    deltas = [
        (row[index] - base[row[0]]) / base[row[0]] * 100.0
        for row in cand_rows
        if row[0] in base and base[row[0]] > 0
    ]
    if not deltas:
        return None
    return sum(deltas) / len(deltas)


def render_svg(base_rows, cand_rows, base_label, cand_label, title, width, height):
    margin_left = 82
    margin_right = 82
    margin_top = 92
    margin_bottom = 72
    plot = (
        margin_left,
        margin_top,
        width - margin_left - margin_right,
        height - margin_top - margin_bottom,
    )
    left, top, plot_width, plot_height = plot
    right = left + plot_width
    bottom = top + plot_height

    all_rows = base_rows + cand_rows
    x_min = 0
    x_max = max(row[0] for row in all_rows)
    prefill_max = nice_ceil(max(row[1] for row in all_rows) * 1.05)
    gen_max = nice_ceil(max(row[2] for row in all_rows) * 1.05)

    x_step = nice_step(x_max - x_min, 6)
    x_ticks = [tick for tick in frange(x_step, x_max, x_step)]
    prefill_ticks = [tick for tick in frange(0, prefill_max, nice_step(prefill_max, 5))]
    gen_ticks = [tick for tick in frange(0, gen_max, nice_step(gen_max, 5))]

    lines = [
        (BASE_PREFILL_COLOR, DASH, [(row[0], row[1]) for row in base_rows], prefill_max),
        (BASE_GEN_COLOR, DASH, [(row[0], row[2]) for row in base_rows], gen_max),
        (PREFILL_COLOR, None, [(row[0], row[1]) for row in cand_rows], prefill_max),
        (GEN_COLOR, None, [(row[0], row[2]) for row in cand_rows], gen_max),
    ]

    deltas = []
    for name, index in (("prefill", 1), ("generation", 2)):
        delta = mean_delta_pct(base_rows, cand_rows, index)
        if delta is not None:
            deltas.append(f"{name} {delta:+.1f}%")
    subtitle = " · ".join(deltas)
    if subtitle:
        subtitle += f" ({cand_label} vs {base_label}, avg over shared ctx)"

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }",
        ".title { font-size: 26px; font-weight: 700; fill: #1f2933; }",
        ".subtitle { font-size: 14px; fill: #64748b; }",
        ".axis-label { font-size: 14px; font-weight: 600; fill: #334155; }",
        ".tick { font-size: 12px; fill: #64748b; }",
        ".legend { font-size: 13px; font-weight: 600; fill: #1f2933; }",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text class="title" x="{width / 2:.1f}" y="34" text-anchor="middle">{html.escape(title)}</text>',
    ]
    if subtitle:
        parts.append(
            f'<text class="subtitle" x="{width / 2:.1f}" y="58" text-anchor="middle">{html.escape(subtitle)}</text>'
        )

    # Horizontal grid and left-axis labels use the prefill scale.
    for tick in prefill_ticks:
        y = bottom - tick / prefill_max * plot_height
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="{GRID_COLOR}" stroke-width="1"/>')
        parts.append(f'<text class="tick" x="{left - 12}" y="{y + 4:.2f}" text-anchor="end">{fmt_tick(tick)}</text>')

    # Right-axis labels use the generation scale.
    for tick in gen_ticks:
        y = bottom - tick / gen_max * plot_height
        parts.append(f'<text class="tick" x="{right + 12}" y="{y + 4:.2f}" text-anchor="start">{fmt_tick(tick)}</text>')

    for tick in x_ticks:
        x = left + (tick - x_min) / (x_max - x_min) * plot_width
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" stroke="{GRID_COLOR}" stroke-width="1"/>')
        parts.append(f'<text class="tick" x="{x:.2f}" y="{bottom + 24}" text-anchor="middle">{fmt_tick(tick)}</text>')

    parts.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="{AXIS_COLOR}" stroke-width="1.4"/>',
            f'<line x1="{right}" y1="{top}" x2="{right}" y2="{bottom}" stroke="{AXIS_COLOR}" stroke-width="1.4"/>',
            f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="{AXIS_COLOR}" stroke-width="1.4"/>',
            f'<text class="axis-label" x="{width / 2:.1f}" y="{height - 20}" text-anchor="middle">ctx size</text>',
            f'<text class="axis-label" x="22" y="{top + plot_height / 2:.1f}" text-anchor="middle" transform="rotate(-90 22 {top + plot_height / 2:.1f})">prefill t/s</text>',
            f'<text class="axis-label" x="{width - 22}" y="{top + plot_height / 2:.1f}" text-anchor="middle" transform="rotate(90 {width - 22} {top + plot_height / 2:.1f})">generation t/s</text>',
        ]
    )

    for color, dash, points, y_max in lines:
        poly = points_to_polyline(points, x_min, x_max, y_max, plot)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"{dash_attr} points="{poly}"/>'
        )

    legend_entries = [
        (BASE_PREFILL_COLOR, DASH, f"prefill — {base_label}"),
        (PREFILL_COLOR, None, f"prefill — {cand_label}"),
        (BASE_GEN_COLOR, DASH, f"generation — {base_label}"),
        (GEN_COLOR, None, f"generation — {cand_label}"),
    ]
    legend_width = 16 + 34 + max(len(text) for _, _, text in legend_entries) * 7
    legend_x = right - legend_width - 6
    legend_y = top + 18
    parts.append(
        f'<rect x="{legend_x - 14}" y="{legend_y - 18}" width="{legend_width + 14}" height="{len(legend_entries) * 26 + 10}" rx="6" fill="#ffffff" fill-opacity="0.92" stroke="#cbd5e1"/>'
    )
    for i, (color, dash, text) in enumerate(legend_entries):
        y = legend_y + i * 26
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<line x1="{legend_x}" y1="{y:.2f}" x2="{legend_x + 26}" y2="{y:.2f}" stroke="{color}" stroke-width="3" stroke-linecap="round"{dash_attr}/>'
        )
        parts.append(f'<text class="legend" x="{legend_x + 34}" y="{y + 4:.2f}">{html.escape(text)}</text>')

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Overlay two ds4-bench CSV files in one SVG graph.")
    parser.add_argument("baseline", type=Path, help="baseline CSV (drawn dashed)")
    parser.add_argument("candidate", type=Path, help="candidate CSV (drawn solid)")
    parser.add_argument("-o", "--output", type=Path, help="output SVG path")
    parser.add_argument("--label-a", help="legend label for the baseline; defaults to its file name")
    parser.add_argument("--label-b", help="legend label for the candidate; defaults to its file name")
    parser.add_argument("--title", help="graph title")
    parser.add_argument("--width", type=int, default=960, help="SVG width in pixels")
    parser.add_argument("--height", type=int, default=560, help="SVG height in pixels")
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = args.candidate.with_name(f"{args.candidate.stem}_vs_{args.baseline.stem}_ts.svg")

    base_rows = read_points(args.baseline)
    cand_rows = read_points(args.candidate)
    base_label = args.label_a or derive_label(args.baseline)
    cand_label = args.label_b or derive_label(args.candidate)
    title = args.title or f"{cand_label} vs {base_label} t/s"
    output.write_text(
        render_svg(base_rows, cand_rows, base_label, cand_label, title, args.width, args.height),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
