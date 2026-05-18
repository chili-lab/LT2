#!/usr/bin/env python3
"""
Plot estimated forward-pass FLOPs vs sequence length for LT2 configs.

Uses Chinchilla-style accounting (2 FLOPs per MAC) for embeddings and logits as in
Appendix F; SwiGLU FFN as three matmuls (6 * s * d * f_eff); dense attention per layer
as 8*s*d^2 + 4*s^2*d + 3*h*s^2 when head_dim = d / n_heads.

GDN (fla GatedDeltaNet as wired in LinearAttentionBlock): projection-dominated estimate
~ 21 * s * d^2 plus depthwise short-conv and a linear-in-s recurrent term (see docstring).

Effective transformer depth is loop_count * n_layers (weight-tied layers are
executed multiple times, so FLOPs scale accordingly).

Default sequence lengths (x-axis): 16k, 32k, …, 1M (powers-of-two ladder from 16384).
Override with ``--s-points``. Curves are drawn with connecting lines plus markers.

Output is a **single row of two panels**: left = attention-path FLOPs only (sum over
effective layer passes); right = full forward pass (embed + layers + logits). White
background, high-contrast palette, per-panel legends in the top-left corner; no
in-figure caption (caption lives in the LaTeX document).

Example:
  python plot_flops_vs_seqlen.py --out flops_vs_seqlen.svg
  python plot_flops_vs_seqlen.py --out plot.png   # needs matplotlib
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

try:
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# Default x-axis starts at 16k (measured lengths only; no interpolation between points).
DEFAULT_SEQ_LENS: List[int] = [
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
]

# ----- Palette: white background, high-contrast series -----
PAPER = "#FFFFFF"
AXIS_COL = "#1A1A1A"
GRID_COL = "#CCCCCC"
TEXT_TITLE = "#111111"
TEXT_BODY = "#222222"
TEXT_MUTED = "#555555"

# Series: GDN looped (green = efficient), Full attn looped (orange), Full attn no-loop (red).
SERIES_FACE = ["#1B9E77", "#D95F02", "#C1272D"]
SERIES_EDGE = ["#0B5E47", "#7A3500", "#6F1618"]
# Distinct marker shapes (matplotlib + SVG).
MARKER_MPL = ["o", "s", "^"]
# Line dash styles to further distinguish series.
LINE_DASH_SVG = ["", "6 4", "2 3"]
LINE_DASH_MPL = ["-", "--", ":"]

# Typography: sans-serif stack for SVG (double-quoted attributes).
FONT_SVG = "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"


def _parse_s_points(text: str) -> List[int]:
    out: List[int] = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p, 10))
    if not out:
        raise ValueError("empty --s-points")
    return out


def _format_seqlen_tick(s: float) -> str:
    """Round to the nearest integer k/M (so 1048576 -> 1M, 32768 -> 33k)."""
    if s >= 1_000_000:
        return f"{int(round(s / 1_000_000))}M"
    if s >= 1000:
        return f"{int(round(s / 1000))}k"
    return str(int(s))


def _svg_datum_marker(idx: int, cx: float, cy: float, fill: str, stroke: str, r: float = 5.5) -> str:
    """One discrete sample marker (no connecting line). Shape varies by series index."""
    sw = 1.25
    if idx % 3 == 0:
        return (
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>'
        )
    if idx % 3 == 1:
        d = 2 * r / math.sqrt(2)
        x0, y0 = cx - d / 2, cy - d / 2
        return (
            f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{d:.2f}" height="{d:.2f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )
    # triangle
    h = 1.25 * r
    w = 1.15 * r
    p1 = f"{cx:.2f},{cy - h:.2f}"
    p2 = f"{cx - w:.2f},{cy + 0.55 * h:.2f}"
    p3 = f"{cx + w:.2f},{cy + 0.55 * h:.2f}"
    return f'<polygon points="{p1} {p2} {p3}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round"/>'


def _svg_legend_marker(idx: int, cx: float, cy: float, fill: str, stroke: str) -> str:
    return _svg_datum_marker(idx, cx, cy, fill, stroke, r=4.0)


@dataclass
class ModelSpec:
    dim: int
    n_layers: int
    n_heads: int
    vocab_size: int
    multiple_of: int
    ffn_dim_multiplier: Optional[float]
    loop_count: int
    layer_kind: str  # "full" | "gdn"


def _ffn_hidden_dim(dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]) -> int:
    """Match lingua/lingua/transformer.py FeedForward."""
    hidden_dim = int(2 * (4 * dim) / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


def _effective_layer_passes(spec: ModelSpec) -> int:
    """How many block forwards run per token embedding → logits forward."""
    return spec.n_layers * spec.loop_count


def _gdn_dims(dim: int, n_heads: int) -> Tuple[int, int, int, int]:
    """Match apps/LT2/transformer.py LinearAttentionBlock + fla GatedDeltaNet defaults."""
    target_key_dim = int(0.75 * dim)
    if target_key_dim % n_heads != 0:
        raise ValueError(f"0.75*dim must divide n_heads (dim={dim}, n_heads={n_heads})")
    head_k_dim = target_key_dim // n_heads
    expand_v = 2.0
    head_v_dim = int(head_k_dim * expand_v)
    key_dim = n_heads * head_k_dim
    value_dim = n_heads * head_v_dim
    return key_dim, value_dim, head_k_dim, head_v_dim


def flops_full_attn_attn_only(s: int, d: int, h: int) -> float:
    """Dense MHA FLOPs only (no FFN), head_dim = d // h, kh = d."""
    k = d // h
    assert h * k == d
    return 8.0 * s * d * d + 4.0 * s * s * d + 3.0 * h * s * s


def flops_full_attn_layer(s: int, d: int, h: int, f: int) -> float:
    """Dense MHA + SwiGLU FFN per block."""
    return flops_full_attn_attn_only(s, d, h) + 6.0 * s * d * f


def flops_gdn_layer(s: int, d: int, h: int, f: int, conv_kernel: int = 4) -> float:
    """
    GatedDeltaNet forward (training, use_short_conv=True, use_gate=True).

    Matmuls (2 FLOPs per MAC): q_proj, k_proj (d->key_dim), v_proj (d->value_dim),
    a_proj, b_proj (d->h), g_proj (d->value_dim), o_proj (value_dim->d).

    Depthwise short convs: ~2 * s * kernel * (2*key_dim + value_dim) (per channel per tap).

    chunk_gated_delta_rule: order O(s * h * head_k * head_v); use 2*s*h*dk*dv as a
    single pass-through MAC-style bound (same order as many linear-attention kernels).
    """
    key_dim, value_dim, head_k_dim, head_v_dim = _gdn_dims(d, h)

    flops = 0.0
    # projections
    flops += 2.0 * s * d * key_dim  # q_proj
    flops += 2.0 * s * d * key_dim  # k_proj
    flops += 2.0 * s * d * value_dim  # v_proj
    flops += 2.0 * s * d * h  # a_proj
    flops += 2.0 * s * d * h  # b_proj
    flops += 2.0 * s * d * value_dim  # g_proj
    flops += 2.0 * s * value_dim * d  # o_proj

    # depthwise causal conv on q/k/v intermediates
    flops += 3 * (2.0 * s * conv_kernel * (2 * key_dim + value_dim))

    # recurrent / chunk core (linear in s, no s^2)
    flops += 2.0 * s * h * head_k_dim * head_v_dim

    # SwiGLU FFN (three linears)
    flops += 6.0 * s * d * f
    return flops


def flops_gdn_attn_only(s: int, d: int, h: int, f: int, conv_kernel: int = 4) -> float:
    """GDN submodule only (no SwiGLU FFN)."""
    return flops_gdn_layer(s, d, h, f, conv_kernel) - 6.0 * s * d * f


def attention_path_flops_per_sequence(spec: ModelSpec, s: int) -> float:
    """Attention-path FLOPs only, summed over effective layer passes (no embed / logits / FFN)."""
    d = spec.dim
    h = spec.n_heads
    f = _ffn_hidden_dim(d, spec.multiple_of, spec.ffn_dim_multiplier)
    eff_layers = _effective_layer_passes(spec)
    if spec.layer_kind == "full":
        per = flops_full_attn_attn_only(s, d, h)
    elif spec.layer_kind == "gdn":
        per = flops_gdn_attn_only(s, d, h, f)
    else:
        raise ValueError(spec.layer_kind)
    return eff_layers * float(per)


def forward_flops_per_sequence(spec: ModelSpec, s: int) -> float:
    """Forward FLOPs for one sequence of length s (embed + stacked blocks + logits)."""
    d = spec.dim
    h = spec.n_heads
    f = _ffn_hidden_dim(d, spec.multiple_of, spec.ffn_dim_multiplier)
    v = spec.vocab_size

    eff_layers = _effective_layer_passes(spec)

    embed = 2.0 * s * v * d
    logits = 2.0 * s * d * v

    if spec.layer_kind == "full":
        layer = flops_full_attn_layer(s, d, h, f)
    elif spec.layer_kind == "gdn":
        layer = flops_gdn_layer(s, d, h, f)
    else:
        raise ValueError(spec.layer_kind)

    return embed + logits + eff_layers * layer


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def spec_from_lt2_yaml(
    path: Path,
    *,
    loop_count: int,
    vocab_size: int,
    layer_kind: Optional[str] = None,
) -> ModelSpec:
    cfg = load_yaml(path)
    m = cfg["model"]
    if layer_kind is None:
        lp = m.get("layer_pattern", "full")
        layer_kind = "gdn" if lp == "gdn" else "full"
    return ModelSpec(
        dim=int(m["dim"]),
        n_layers=int(m["n_layers"]),
        n_heads=int(m["n_heads"]),
        vocab_size=int(vocab_size),
        multiple_of=int(m.get("multiple_of", 256)),
        ffn_dim_multiplier=m.get("ffn_dim_multiplier"),
        loop_count=int(loop_count),
        layer_kind=layer_kind,
    )


def _write_svg_two_panel_row(
    path: Path,
    s_vals: Sequence[float],
    series_attn: List[Tuple[str, List[float]]],
    series_e2e: List[Tuple[str, List[float]]],
    *,
    log_x: bool,
) -> None:
    """
    One row, two panels (each ~half the total width, overall ~3:2 aspect).
    White background, high-contrast series, lines connecting markers, legend top-left
    in each panel, no in-figure caption (caption lives in the LaTeX document).
    """
    w, h = 1200, 800
    paper = PAPER
    axis_col = AXIS_COL
    grid_col = GRID_COL

    panel_y0 = 24.0
    panel_h = h - panel_y0 - 24.0
    gap = 40.0
    outer = 24.0
    plot_w_each = (w - outer * 2 - gap) / 2.0

    def panel_blocks(
        px: float,
        series: List[Tuple[str, List[float]]],
        y_axis_short: str,
    ) -> str:
        margin_l, margin_r, margin_t, margin_b = 92.0, 24.0, 28.0, 104.0
        plot_w = plot_w_each - margin_l - margin_r
        plot_h = panel_h - margin_t - margin_b
        plot_left = px + margin_l
        plot_top = panel_y0 + margin_t

        all_y = [y for _, ys in series for y in ys if y > 0]
        y_min, y_max = min(all_y), max(all_y)
        lo = 10 ** (math.floor(math.log10(y_min)) - 0.05)
        hi = 10 ** (math.ceil(math.log10(y_max)) + 0.05)

        s_lo, s_hi = float(min(s_vals)), float(max(s_vals))
        # Extra horizontal padding so tick labels don't crowd each other.
        pad_frac = 0.04
        if log_x:
            lx0, lx1 = math.log10(s_lo), math.log10(s_hi)
            lspan = lx1 - lx0

            def x_px(sv: float) -> float:
                return plot_left + pad_frac * plot_w + (
                    (math.log10(sv) - lx0) / lspan
                ) * plot_w * (1 - 2 * pad_frac)
        else:
            span = s_hi - s_lo

            def x_px(sv: float) -> float:
                return plot_left + pad_frac * plot_w + (
                    (sv - s_lo) / span
                ) * plot_w * (1 - 2 * pad_frac)

        def y_px(fl: float) -> float:
            return plot_top + (
                1.0 - (math.log10(fl) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
            ) * plot_h

        parts: List[str] = []

        # Horizontal gridlines at decades.
        k = math.floor(math.log10(lo))
        k_end = math.ceil(math.log10(hi))
        while k <= k_end:
            tv = 10**k
            if lo <= tv <= hi:
                yy = y_px(tv)
                parts.append(
                    f'<line x1="{plot_left:.2f}" y1="{yy:.2f}" x2="{plot_left + plot_w:.2f}" y2="{yy:.2f}" '
                    f'stroke="{grid_col}" stroke-width="1" stroke-dasharray="4 4"/>'
                )
            k += 1

        # Series: connecting line then markers on top.
        for idx, (_lab, ys) in enumerate(series):
            fc = SERIES_FACE[idx % len(SERIES_FACE)]
            ec = SERIES_EDGE[idx % len(SERIES_EDGE)]
            dash = LINE_DASH_SVG[idx % len(LINE_DASH_SVG)]
            pts = " ".join(f"{x_px(float(sv)):.2f},{y_px(yv):.2f}" for sv, yv in zip(s_vals, ys))
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            parts.append(
                f'<polyline points="{pts}" fill="none" stroke="{ec}" stroke-width="2.2"'
                f'{dash_attr} stroke-linejoin="round" stroke-linecap="round"/>'
            )
            for sv, yv in zip(s_vals, ys):
                parts.append(_svg_datum_marker(idx, x_px(float(sv)), y_px(yv), fc, ec, r=6.0))

        # Frame (axes).
        parts.append(
            f'<line x1="{plot_left:.2f}" y1="{plot_top + plot_h:.2f}" x2="{plot_left + plot_w:.2f}" '
            f'y2="{plot_top + plot_h:.2f}" stroke="{axis_col}" stroke-width="1.8"/>'
            f'<line x1="{plot_left:.2f}" y1="{plot_top:.2f}" x2="{plot_left:.2f}" '
            f'y2="{plot_top + plot_h:.2f}" stroke="{axis_col}" stroke-width="1.8"/>'
        )

        # X ticks (rotated labels, generously spaced).
        for sv in s_vals:
            xx = x_px(float(sv))
            parts.append(
                f'<line x1="{xx:.2f}" y1="{plot_top + plot_h:.2f}" x2="{xx:.2f}" '
                f'y2="{plot_top + plot_h + 6:.2f}" stroke="{axis_col}" stroke-width="1.2"/>'
                f'<text transform="translate({xx:.2f},{plot_top + plot_h + 26:.2f}) rotate(-40)" '
                f'text-anchor="end" font-family="{FONT_SVG}" font-size="14" fill="{TEXT_BODY}">'
                f'{_svg_escape(_format_seqlen_tick(float(sv)))}</text>'
            )

        # Y ticks.
        k = math.floor(math.log10(lo))
        k_end = math.ceil(math.log10(hi))
        while k <= k_end:
            tv = 10**k
            if lo <= tv <= hi:
                yy = y_px(tv)
                parts.append(
                    f'<line x1="{plot_left - 5:.2f}" y1="{yy:.1f}" x2="{plot_left:.2f}" y2="{yy:.1f}" '
                    f'stroke="{axis_col}"/>'
                    f'<text x="{plot_left - 10:.2f}" y="{yy + 4.5:.1f}" text-anchor="end" '
                    f'font-family="{FONT_SVG}" font-size="13" fill="{TEXT_BODY}">10<tspan font-size="10" baseline-shift="super">{k}</tspan></text>'
                )
            k += 1

        # Y-axis label.
        mid_y = plot_top + plot_h / 2.0
        y_lab_x = plot_left - 66.0
        parts.append(
            f'<text transform="translate({y_lab_x:.1f},{mid_y:.1f}) rotate(-90)" '
            f'text-anchor="middle" font-family="{FONT_SVG}" font-size="15" font-weight="700" '
            f'fill="{TEXT_TITLE}">{_svg_escape(y_axis_short)}</text>'
        )

        # X-axis label (per panel).
        parts.append(
            f'<text x="{plot_left + plot_w / 2:.1f}" y="{plot_top + plot_h + 84:.1f}" '
            f'text-anchor="middle" font-family="{FONT_SVG}" font-size="15" font-weight="700" '
            f'fill="{TEXT_TITLE}">Sequence length</text>'
        )

        # Legend (top-left, inside plot area).
        leg_x = plot_left + 14.0
        leg_y = plot_top + 14.0
        row_h = 22.0
        labels_here = [lab for lab, _ in series]
        max_chars = max(len(lab) for lab in labels_here)
        box_w = 30.0 + 7.2 * max_chars
        box_h = row_h * len(labels_here) + 10.0
        parts.append(
            f'<rect x="{leg_x:.1f}" y="{leg_y:.1f}" width="{box_w:.1f}" height="{box_h:.1f}" '
            f'fill="{PAPER}" fill-opacity="0.92" stroke="{axis_col}" stroke-width="1"/>'
        )
        for idx, lab in enumerate(labels_here):
            fc = SERIES_FACE[idx % len(SERIES_FACE)]
            ec = SERIES_EDGE[idx % len(SERIES_EDGE)]
            dash = LINE_DASH_SVG[idx % len(LINE_DASH_SVG)]
            cy = leg_y + 16 + idx * row_h
            lx1 = leg_x + 8.0
            lx2 = leg_x + 28.0
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            parts.append(
                f'<line x1="{lx1:.1f}" y1="{cy:.1f}" x2="{lx2:.1f}" y2="{cy:.1f}" '
                f'stroke="{ec}" stroke-width="2.2"{dash_attr} stroke-linecap="round"/>'
            )
            parts.append(_svg_legend_marker(idx, (lx1 + lx2) / 2, cy, fc, ec))
            parts.append(
                f'<text x="{lx2 + 6:.1f}" y="{cy + 4:.1f}" font-family="{FONT_SVG}" '
                f'font-size="13" font-weight="500" fill="{TEXT_BODY}">{_svg_escape(lab)}</text>'
            )

        return "".join(parts)

    left_x = outer
    right_x = outer + plot_w_each + gap
    left_svg = panel_blocks(left_x, series_attn, "Attention FLOPs / sequence")
    right_svg = panel_blocks(right_x, series_e2e, "Forward FLOPs / sequence")

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <rect width="100%" height="100%" fill="{paper}"/>
  {left_svg}
  {right_svg}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def _svg_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def resolve_vocab_size(cfg_path: Path, override: Optional[int]) -> int:
    if override is not None:
        return int(override)
    cfg = load_yaml(cfg_path)
    tok = cfg.get("data", {}).get("tokenizer", {})
    if tok.get("name") == "tiktoken":
        try:
            import tiktoken

            return tiktoken.get_encoding("cl100k_base").n_vocab
        except Exception:
            pass
    return 100277  # cl100k_base–scale default if unknown


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--gdn-yaml",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "1B"
        / "looped_1B_pure_gdn_neg_nminus1_loop.yaml",
    )
    p.add_argument(
        "--full-yaml",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "1B"
        / "looped_1B_full_attn.yaml",
    )
    p.add_argument("--loop-count", type=int, default=8, help="loop_count for both looped curves")
    p.add_argument(
        "--s-points",
        type=str,
        default=None,
        help=(
            "Comma-separated sequence lengths (e.g. 1024,2048,4096). "
            f"Default: built-in ladder 16k→1M ({len(DEFAULT_SEQ_LENS)} points)."
        ),
    )
    p.add_argument(
        "--linear-x",
        action="store_true",
        help="Use linear x-axis mapping (default is log x when span exceeds ~32×).",
    )
    p.add_argument("--vocab-size", type=int, default=None, help="override vocab V in FLOP formulas")
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "flops_vs_seqlen.svg",
        help="Output path (.png/.pdf use matplotlib if installed; otherwise .svg)",
    )
    args = p.parse_args()

    vocab = resolve_vocab_size(args.full_yaml, args.vocab_size)

    spec_gdn_looped = spec_from_lt2_yaml(args.gdn_yaml, loop_count=args.loop_count, vocab_size=vocab)
    spec_full_looped = spec_from_lt2_yaml(args.full_yaml, loop_count=args.loop_count, vocab_size=vocab)
    spec_full_no_loop = spec_from_lt2_yaml(args.full_yaml, loop_count=1, vocab_size=vocab)

    s_int = _parse_s_points(args.s_points) if args.s_points else list(DEFAULT_SEQ_LENS)
    s_vals = [float(s) for s in s_int]
    span_ratio = max(s_int) / max(1, min(s_int))
    log_x = (not args.linear_x) and span_ratio > 32

    y_gdn_e2e = [forward_flops_per_sequence(spec_gdn_looped, s) for s in s_int]
    y_full_looped_e2e = [forward_flops_per_sequence(spec_full_looped, s) for s in s_int]
    y_full_single_e2e = [forward_flops_per_sequence(spec_full_no_loop, s) for s in s_int]

    y_gdn_attn = [attention_path_flops_per_sequence(spec_gdn_looped, s) for s in s_int]
    y_full_looped_attn = [attention_path_flops_per_sequence(spec_full_looped, s) for s in s_int]
    y_full_single_attn = [attention_path_flops_per_sequence(spec_full_no_loop, s) for s in s_int]

    labels = [
        "Looped linear attention (ours)",
        "Looped full attention",
        "Standard Transformer (full attention)",
    ]
    series_attn = list(zip(labels, [y_gdn_attn, y_full_looped_attn, y_full_single_attn]))
    series_e2e = list(zip(labels, [y_gdn_e2e, y_full_looped_e2e, y_full_single_e2e]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    suffix = args.out.suffix.lower()
    use_mpl = _HAS_MPL and suffix in (".png", ".pdf")

    if use_mpl:
        plt.rcParams.update(
            {
                "font.size": 13,
                "axes.titlesize": 14,
                "axes.labelsize": 14,
                "legend.fontsize": 12,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "axes.linewidth": 1.4,
            }
        )
        # Overall 3:2 aspect; each panel takes half the space.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 8.0), dpi=150)
        fig.patch.set_facecolor(PAPER)
        for ax in (ax1, ax2):
            ax.set_facecolor(PAPER)
            for spine in ax.spines.values():
                spine.set_color(AXIS_COL)
            ax.tick_params(colors=TEXT_BODY)
            ax.grid(True, which="major", axis="y", ls="--", alpha=0.55, color=GRID_COL)

        attn_ys = [y_gdn_attn, y_full_looped_attn, y_full_single_attn]
        e2e_ys = [y_gdn_e2e, y_full_looped_e2e, y_full_single_e2e]
        handles_per_ax: List[list] = [[], []]
        for i in range(3):
            for ax, ys, store in ((ax1, attn_ys[i], handles_per_ax[0]), (ax2, e2e_ys[i], handles_per_ax[1])):
                (h_,) = ax.plot(
                    s_vals,
                    ys,
                    linestyle=LINE_DASH_MPL[i],
                    linewidth=2.2,
                    color=SERIES_EDGE[i],
                    marker=MARKER_MPL[i],
                    markersize=10,
                    markerfacecolor=SERIES_FACE[i],
                    markeredgecolor=SERIES_EDGE[i],
                    markeredgewidth=1.3,
                    label=labels[i],
                )
                store.append(h_)

        for ax in (ax1, ax2):
            if log_x:
                ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(s_int)
            ax.set_xticklabels(
                [_format_seqlen_tick(float(s)) for s in s_int], rotation=40, ha="right"
            )
            # Add a little horizontal padding so tick labels breathe.
            if log_x:
                s_lo, s_hi = min(s_int), max(s_int)
                lx0, lx1 = math.log10(s_lo), math.log10(s_hi)
                pad = 0.04 * (lx1 - lx0)
                ax.set_xlim(10 ** (lx0 - pad), 10 ** (lx1 + pad))
            ax.set_xlabel("Sequence length", color=TEXT_TITLE, fontweight="bold")

        ax1.set_ylabel("Attention FLOPs / sequence", color=TEXT_TITLE, fontweight="bold")
        ax2.set_ylabel("Forward FLOPs / sequence", color=TEXT_TITLE, fontweight="bold")

        for ax, hs in zip((ax1, ax2), handles_per_ax):
            leg = ax.legend(
                handles=hs,
                loc="upper left",
                frameon=True,
                fancybox=False,
                edgecolor=AXIS_COL,
                facecolor=PAPER,
                framealpha=0.92,
            )
            plt.setp(leg.get_texts(), color=TEXT_BODY)

        fig.tight_layout()
        fig.savefig(args.out, facecolor=fig.get_facecolor())
    else:
        svg_path = args.out if suffix == ".svg" else args.out.with_suffix(".svg")
        _write_svg_two_panel_row(
            svg_path,
            s_vals,
            series_attn,
            series_e2e,
            log_x=log_x,
        )
        print(f"Wrote {svg_path.resolve()}")

        # When the user asked for a non-SVG format and matplotlib is unavailable,
        # fall back to cairosvg to produce the requested raster/PDF output.
        if suffix in (".pdf", ".png") and args.out != svg_path:
            try:
                import cairosvg  # type: ignore
            except ImportError:
                print(
                    f"matplotlib and cairosvg both unavailable; only SVG written. "
                    f"Install cairosvg (`pip install cairosvg`) to emit {suffix}."
                )
                return
            args.out.parent.mkdir(parents=True, exist_ok=True)
            if suffix == ".pdf":
                cairosvg.svg2pdf(url=str(svg_path), write_to=str(args.out))
            else:
                cairosvg.svg2png(url=str(svg_path), write_to=str(args.out), output_width=2400)
            print(f"Wrote {args.out.resolve()}")
        return

    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
