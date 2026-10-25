#!/usr/bin/env python3
"""Render assets/pose-library/poses.json into one SVG contact sheet.

Pure standard library. The templates are normalised image coordinates (y down);
each panel rescales the figure to a fixed box. Output:
assets/pose-library/diagrams/pose-library.svg. Regenerate after every pose batch.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "assets" / "pose-library" / "poses.json"
OUT = ROOT / "assets" / "pose-library" / "diagrams" / "pose-library.svg"

EDGES = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]

PANEL_W, PANEL_H = 220, 260
COLS = 4
MARGIN = 18
HEADER = 54


def render_pose(pose, col, row):
    j = {int(k): v for k, v in pose["template"].items()}
    xs = [p[0] for p in j.values()]
    ys = [p[1] for p in j.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    box_w = PANEL_W - 2 * MARGIN
    box_h = PANEL_H - HEADER - MARGIN
    scale = min(box_w / span_x, box_h / span_y)

    def px(idx):
        x, y = j[idx]
        return (
            MARGIN + (box_w - span_x * scale) / 2 + (x - min_x) * scale + col * PANEL_W,
            HEADER + (box_h - span_y * scale) / 2 + (y - min_y) * scale + row * PANEL_H,
        )

    parts = []
    ox, oy = col * PANEL_W, row * PANEL_H
    parts.append(
        f'<rect x="{ox + 1}" y="{oy + 1}" width="{PANEL_W - 2}" height="{PANEL_H - 2}" '
        f'rx="12" fill="#111418" stroke="#2a2f37"/>'
    )
    parts.append(
        f'<text x="{ox + PANEL_W / 2}" y="{oy + 24}" text-anchor="middle" '
        f'fill="#f2f4f7" font-family="sans-serif" font-size="15">{pose["title_zh"]}</text>'
    )
    parts.append(
        f'<text x="{ox + PANEL_W / 2}" y="{oy + 42}" text-anchor="middle" '
        f'fill="#8b94a3" font-family="sans-serif" font-size="10">{pose["title_en"]}</text>'
    )
    for a, b in EDGES:
        x1, y1 = px(a)
        x2, y2 = px(b)
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#ffb36b" stroke-width="2.5" stroke-linecap="round"/>'
        )
    # Nose to shoulder-mid head line.
    nx, ny = px(0)
    sx1, sy1 = px(11)
    sx2, sy2 = px(12)
    parts.append(
        f'<line x1="{nx:.1f}" y1="{ny:.1f}" x2="{(sx1 + sx2) / 2:.1f}" y2="{(sy1 + sy2) / 2:.1f}" '
        f'stroke="#ffb36b" stroke-width="2.5" stroke-linecap="round"/>'
    )
    for idx in (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28):
        x, y = px(idx)
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="#111418" stroke="#ffb36b" stroke-width="1.6"/>'
        )
    return "\n".join(parts)


def main():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    poses = data["poses"]
    rows = (len(poses) + COLS - 1) // COLS
    width = COLS * PANEL_W
    height = rows * PANEL_H + 34
    panels = [
        render_pose(pose, i % COLS, i // COLS)
        for i, pose in enumerate(poses)
    ]
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'<rect width="100%" height="100%" fill="#0c0e12"/>\n'
        f'<text x="{width / 2}" y="22" text-anchor="middle" fill="#8b94a3" '
        f'font-family="sans-serif" font-size="11">途帧 pose library · {len(poses)} standing-portrait templates · CC0-1.0 · normalised template coordinates</text>\n'
        + "\n".join(panels)
        + "\n</svg>\n"
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(svg, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({width}x{height})")


if __name__ == "__main__":
    main()
