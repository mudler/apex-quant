#!/usr/bin/env python3
"""
plot_benchmarks.py -- Generate publication-quality comparison charts from
benchmark JSON files for APEX quantization evaluation.

Produces seven plots:
  1. pareto_ppl_size.png      -- Perplexity vs Model Size scatter
  2. pareto_ppl_speed.png     -- Perplexity vs Speed scatter
  3. radar_chart.png          -- Radar/spider overlay chart
  4. accuracy_comparison.png  -- Grouped bar chart for accuracy benchmarks
  5. kl_comparison.png        -- KL Divergence grouped bars (log scale)
  6. efficiency.png           -- Size vs Speed bubble chart (PPL as bubble size)
  7. kl_apex_vs_unsloth.png   -- APEX vs Unsloth KL divergence comparison

Usage:
  python scripts/plot_benchmarks.py
  python scripts/plot_benchmarks.py --input-dir benchmark_results/final/ --output-dir plots/
"""

import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from adjustText import adjust_text


# ---------------------------------------------------------------------------
# Friendly display names -- map JSON model field to plot label
# ---------------------------------------------------------------------------
DISPLAY_NAMES = {
    "f16":                                  "F16 (64.6 GB)",
    "q8_0":                                 "Q8_0 (34.4 GB)",
    "Qwen3.5-35B-A3B-APEX-I-Quality":      "APEX I-Quality (21.3 GB)",
    "Qwen3.5-35B-A3B-APEX-I-Balanced":     "APEX I-Balanced (23.6 GB)",
    "Qwen3.5-35B-A3B-APEX-I-Compact":      "APEX I-Compact (16.1 GB)",
    "Qwen3.5-35B-A3B-APEX-Mini":           "APEX Mini (12.2 GB)",
    "unsloth_q8_k_xl":                      "Unsloth Q8_K_XL (45.3 GB)",
    "Qwen3.5-35B-A3B-UD-Q4_K_L":           "Unsloth Q4_K_L (18.8 GB)",
    "Qwen_Qwen3.5-35B-A3B-IQ2_M":          "bartowski IQ2_M (11.3 GB)",
    "Qwen_Qwen3.5-35B-A3B-Q3_K_M":         "bartowski Q3_K_M (15.1 GB)",
    "Qwen3.5-35B-A3B-UD-Q4_K_XL":          "Unsloth UD-Q4_K_XL (20.7 GB)",
    "Qwen3.5-35B-A3B-Q5_K_S":              "Q5_K_S (23.1 GB)",
}

# Models to include in plots -- skip any model key not listed here
ALLOWED_MODELS = set(DISPLAY_NAMES.keys())

# ---------------------------------------------------------------------------
# Category classification
# ---------------------------------------------------------------------------
CATEGORY_MAP = {
    "f16":                                  "baseline",
    "q8_0":                                 "baseline",
    "unsloth_q8_k_xl":                      "external",
    "Qwen3.5-35B-A3B-UD-Q4_K_L":           "external",
    "Qwen_Qwen3.5-35B-A3B-IQ2_M":          "external",
    "Qwen_Qwen3.5-35B-A3B-Q3_K_M":         "external",
    "Qwen3.5-35B-A3B-UD-Q4_K_XL":          "external",
    "Qwen3.5-35B-A3B-Q5_K_S":              "baseline",
    "Qwen3.5-35B-A3B-APEX-I-Quality":      "apex",
    "Qwen3.5-35B-A3B-APEX-I-Balanced":     "apex",
    "Qwen3.5-35B-A3B-APEX-I-Compact":      "apex",
    "Qwen3.5-35B-A3B-APEX-Mini":           "apex",
}

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
COLORS = {
    "baseline": "#D32F2F",   # red
    "external": "#F57C00",   # orange
    "apex":     "#388E3C",   # green
}

MARKERS = {
    "baseline": "s",
    "external": "D",
    "apex":     "o",
}

CATEGORY_LABELS = {
    "baseline": "Baselines (F16, Q8_0)",
    "external": "External (Unsloth, bartowski)",
    "apex":     "APEX Configurations",
}

# Per-model colors for charts where every model needs its own color
MODEL_COLORS = {
    "F16 (64.6 GB)":                "#9E9E9E",
    "Q8_0 (34.4 GB)":              "#D32F2F",
    "APEX I-Quality (21.3 GB)":     "#2E7D32",
    "APEX I-Balanced (23.6 GB)":    "#388E3C",
    "APEX I-Compact (16.1 GB)":     "#66BB6A",
    "APEX Mini (12.2 GB)":          "#81C784",
    "Unsloth Q8_K_XL (45.3 GB)":   "#F57C00",
    "Unsloth Q4_K_L (18.8 GB)":    "#FF9800",
    "bartowski IQ2_M (11.3 GB)":    "#7B1FA2",
    "bartowski Q3_K_M (15.1 GB)":   "#AB47BC",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_benchmarks(input_dir):
    """Load all JSON benchmark files from input_dir.

    Returns a list of dicts, each containing the full benchmark data plus
    a 'display_name' and 'category' field.
    """
    models = []
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(input_dir, fname)
        with open(path) as fh:
            data = json.load(fh)
        model_key = data.get("model", fname.replace(".json", ""))
        # Skip models not in the allowed set
        if model_key not in ALLOWED_MODELS:
            print(f"  [skip] {model_key} (not in ALLOWED_MODELS)")
            continue
        data["display_name"] = DISPLAY_NAMES[model_key]
        data["category"] = CATEGORY_MAP.get(model_key, "apex")
        models.append(data)
    if not models:
        print(f"ERROR: No JSON files found in {input_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(models)} benchmark files from {input_dir}")
    for m in models:
        print(f"  - {m['display_name']} (PPL={m['perplexity']})")
    return models


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _apply_style(ax, title, xlabel, ylabel):
    """Apply consistent professional styling."""
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.tick_params(labelsize=10)


def _annotate_point(ax, x, y, label, placed, x_range, y_range):
    """Annotate a scatter point with collision-avoiding offset."""
    dx = x_range * 0.02
    dy = y_range * 0.03
    ox, oy = dx, dy
    for px, py in placed:
        if abs(x - px) < x_range * 0.08 and abs(y - py) < y_range * 0.08:
            oy += dy * 1.5
    ax.annotate(
        label, (x, y),
        textcoords="offset points",
        xytext=(ox * 30, oy * 15),
        fontsize=8,
        ha="left",
        arrowprops=dict(arrowstyle="-", color="#BDBDBD", lw=0.6),
    )
    placed.append((x, y))


# ---------------------------------------------------------------------------
# Plot 1: Perplexity vs Model Size
# ---------------------------------------------------------------------------

def plot_pareto_ppl_size(models, output_dir):
    """Scatter: X=Size (GB), Y=Perplexity. Each point colored by category."""
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    legend_cats = set()
    texts = []

    # Find Q8_0 PPL for reference line
    q8_ppl = None
    for m in models:
        if m["model"] == "q8_0":
            q8_ppl = m["perplexity"]

    for m in models:
        cat = m["category"]
        color = COLORS[cat]
        marker = MARKERS[cat]
        label = CATEGORY_LABELS[cat] if cat not in legend_cats else None
        legend_cats.add(cat)

        ax.scatter(
            m["size_gb"], m["perplexity"],
            color=color, marker=marker, s=100, zorder=5,
            edgecolors="white", linewidths=0.6, label=label,
        )
        texts.append(ax.text(m["size_gb"], m["perplexity"],
                             m["display_name"], fontsize=8))

    # Use adjustText to reposition overlapping labels
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    # Q8_0 reference line
    if q8_ppl is not None:
        ax.axhline(y=q8_ppl, color="#757575", linestyle="--",
                   linewidth=1.0, alpha=0.7)
        ax.annotate(
            f"Q8_0 baseline ({q8_ppl:.4f})",
            xy=(ax.get_xlim()[1], q8_ppl),
            xytext=(-8, 6), textcoords="offset points",
            fontsize=8, color="#757575", ha="right", va="bottom",
        )

    _apply_style(ax, "Perplexity vs Model Size",
                 "Model Size (GB)", "Perplexity (lower is better)")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    fig.tight_layout()

    out = os.path.join(output_dir, "pareto_ppl_size.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Plot 2: Perplexity vs Speed
# ---------------------------------------------------------------------------

def plot_pareto_ppl_speed(models, output_dir):
    """Scatter: X=Speed tg128 (t/s), Y=Perplexity."""
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    legend_cats = set()
    texts = []

    q8_ppl = None
    for m in models:
        if m["model"] == "q8_0":
            q8_ppl = m["perplexity"]

    for m in models:
        cat = m["category"]
        color = COLORS[cat]
        marker = MARKERS[cat]
        label = CATEGORY_LABELS[cat] if cat not in legend_cats else None
        legend_cats.add(cat)

        ax.scatter(
            m["tg128_ts"], m["perplexity"],
            color=color, marker=marker, s=100, zorder=5,
            edgecolors="white", linewidths=0.6, label=label,
        )
        texts.append(ax.text(m["tg128_ts"], m["perplexity"],
                             m["display_name"], fontsize=8))

    # Use adjustText to reposition overlapping labels
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    if q8_ppl is not None:
        ax.axhline(y=q8_ppl, color="#757575", linestyle="--",
                   linewidth=1.0, alpha=0.7)
        ax.annotate(
            f"Q8_0 baseline ({q8_ppl:.4f})",
            xy=(ax.get_xlim()[1], q8_ppl),
            xytext=(-8, 6), textcoords="offset points",
            fontsize=8, color="#757575", ha="right", va="bottom",
        )

    _apply_style(ax, "Perplexity vs Inference Speed",
                 "Speed tg128 (tokens/sec, higher is better)",
                 "Perplexity (lower is better)")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    fig.tight_layout()

    out = os.path.join(output_dir, "pareto_ppl_speed.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Plot 3: Radar / Spider chart
# ---------------------------------------------------------------------------

def plot_radar_chart(models, output_dir):
    """Overlay radar chart comparing selected models across multiple axes."""
    # Models to include in radar
    radar_names = [
        "APEX I-Quality (21.3 GB)", "APEX I-Balanced (23.6 GB)",
        "APEX I-Compact (16.1 GB)", "APEX Mini (12.2 GB)",
        "Unsloth Q8_K_XL (45.3 GB)", "Unsloth Q4_K_L (18.8 GB)",
        "Q8_0 (34.4 GB)",
    ]
    radar_models = [m for m in models if m["display_name"] in radar_names]
    radar_models.sort(key=lambda m: radar_names.index(m["display_name"]))

    if len(radar_models) < 2:
        print("  [!] Not enough models for radar chart, skipping.")
        return None

    # Axes: PPL (inverted+normalized), HellaSwag, MMLU, ARC, Speed, Size efficiency
    axis_labels = [
        "PPL\n(inverted)",
        "HellaSwag",
        "MMLU",
        "ARC\nChallenge",
        "Speed\n(tg128)",
        "Size\nEfficiency",
    ]

    # Extract raw values
    raw_data = {}
    for m in radar_models:
        name = m["display_name"]
        raw_data[name] = {
            "ppl_inv": 1.0 / m["perplexity"],   # inverted: higher = better
            "hellaswag": m["hellaswag"],
            "mmlu": m["mmlu"],
            "arc": m["arc_challenge"],
            "speed": m["tg128_ts"],
            "size_eff": 1.0 / m["size_gb"],      # higher = more compact
        }

    # Normalize each axis to 0-1 range across the included models
    keys = ["ppl_inv", "hellaswag", "mmlu", "arc", "speed", "size_eff"]
    mins = {}
    maxs = {}
    for k in keys:
        vals = [raw_data[n][k] for n in raw_data]
        mins[k] = min(vals)
        maxs[k] = max(vals)

    def normalize(val, key):
        r = maxs[key] - mins[key]
        if r == 0:
            return 0.5
        return (val - mins[key]) / r

    # Build normalized data arrays
    norm_data = {}
    for name in raw_data:
        norm_data[name] = [normalize(raw_data[name][k], k) for k in keys]

    # Radar plot
    n_axes = len(axis_labels)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")

    radar_colors = {
        "APEX I-Quality (21.3 GB)":     "#2E7D32",
        "APEX I-Balanced (23.6 GB)":    "#388E3C",
        "APEX I-Compact (16.1 GB)":     "#66BB6A",
        "APEX Mini (12.2 GB)":          "#81C784",
        "Unsloth Q8_K_XL (45.3 GB)":   "#F57C00",
        "Unsloth Q4_K_L (18.8 GB)":    "#FF9800",
        "Q8_0 (34.4 GB)":              "#D32F2F",
    }
    radar_linestyles = {
        "APEX I-Quality (21.3 GB)":     "-",
        "APEX I-Balanced (23.6 GB)":    "--",
        "APEX I-Compact (16.1 GB)":     "-.",
        "APEX Mini (12.2 GB)":          ":",
        "Unsloth Q8_K_XL (45.3 GB)":   "-",
        "Unsloth Q4_K_L (18.8 GB)":    ":",
        "Q8_0 (34.4 GB)":              "--",
    }

    for name in radar_names:
        if name not in norm_data:
            continue
        values = norm_data[name] + norm_data[name][:1]  # close
        color = radar_colors.get(name, "#999999")
        ls = radar_linestyles.get(name, "-")
        ax.plot(angles, values, color=color, linewidth=2.0, linestyle=ls,
                label=name)
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.set_thetagrids(np.degrees(angles[:-1]), axis_labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8,
                       color="#757575")
    ax.set_title("Multi-Metric Comparison (Normalized)",
                 fontsize=14, fontweight="bold", pad=24)
    ax.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1.25, 1.12),
              framealpha=0.9)

    fig.tight_layout()
    out = os.path.join(output_dir, "radar_chart.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Plot 4: Accuracy comparison grouped bar chart
# ---------------------------------------------------------------------------

def plot_accuracy_comparison(models, output_dir):
    """Grouped bar chart for accuracy benchmarks across all models."""
    # Order of models in the chart
    model_order = [
        "F16 (64.6 GB)", "Q8_0 (34.4 GB)",
        "APEX I-Quality (21.3 GB)", "APEX I-Balanced (23.6 GB)",
        "APEX I-Compact (16.1 GB)", "APEX Mini (12.2 GB)",
        "Unsloth Q8_K_XL (45.3 GB)", "Unsloth Q4_K_L (18.8 GB)",
        "bartowski IQ2_M (11.3 GB)", "bartowski Q3_K_M (15.1 GB)",
    ]
    ordered = []
    for name in model_order:
        for m in models:
            if m["display_name"] == name:
                ordered.append(m)
                break

    if not ordered:
        print("  [!] No models found for accuracy comparison, skipping.")
        return None

    benchmarks = [
        ("HellaSwag", "hellaswag"),
        ("Winogrande", "winogrande"),
        ("MMLU", "mmlu"),
        ("ARC-Challenge", "arc_challenge"),
        ("TruthfulQA", "truthfulqa"),
    ]

    n_models = len(ordered)
    n_benchmarks = len(benchmarks)
    bar_width = 0.12
    group_gap = 0.25

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    group_centers = []
    for gi, (bench_label, bench_key) in enumerate(benchmarks):
        group_start = gi * (n_models * bar_width + group_gap)
        group_center = group_start + (n_models - 1) * bar_width / 2
        group_centers.append(group_center)

        for mi, m in enumerate(ordered):
            val = m.get(bench_key, 0) or 0
            x = group_start + mi * bar_width
            color = MODEL_COLORS.get(m["display_name"], "#999999")
            ax.bar(x, val, width=bar_width * 0.88, color=color,
                   edgecolor="white", linewidth=0.5)
            # Value label
            ax.text(x, val + 0.3, f"{val:.1f}",
                    ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(group_centers)
    ax.set_xticklabels([b[0] for b in benchmarks], fontsize=11)

    # Legend
    handles = []
    for m in ordered:
        color = MODEL_COLORS.get(m["display_name"], "#999999")
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=color,
                       edgecolor="white", linewidth=0.5,
                       label=m["display_name"]))
    ax.legend(handles=handles, fontsize=9, loc="lower right",
              ncol=2, framealpha=0.9)

    # Y axis range: start a bit below the minimum value for clarity
    all_vals = []
    for m in ordered:
        for _, key in benchmarks:
            v = m.get(key, 0) or 0
            if v > 0:
                all_vals.append(v)
    if all_vals:
        y_min = min(all_vals) - 3
        y_max = max(all_vals) + 4
        ax.set_ylim(max(0, y_min), y_max)

    _apply_style(ax, "Accuracy Benchmark Comparison",
                 "", "Accuracy (%)")
    fig.tight_layout()

    out = os.path.join(output_dir, "accuracy_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Plot 5: KL Divergence comparison
# ---------------------------------------------------------------------------

def plot_kl_comparison(models, output_dir):
    """Grouped bars for KL Mean and KL 99.9% (log scale Y axis)."""
    # Only models with KL data
    kl_models = [m for m in models
                 if m.get("kl_mean") is not None and m["kl_mean"] is not None]

    if not kl_models:
        print("  [!] No KL divergence data found, skipping.")
        return None

    # Sort by KL mean (ascending)
    kl_order = [
        "Q8_0 (34.4 GB)",
        "APEX I-Quality (21.3 GB)", "APEX I-Balanced (23.6 GB)",
        "APEX I-Compact (16.1 GB)", "APEX Mini (12.2 GB)",
        "Unsloth Q4_K_L (18.8 GB)",
        "bartowski IQ2_M (11.3 GB)", "bartowski Q3_K_M (15.1 GB)",
    ]
    ordered = []
    for name in kl_order:
        for m in kl_models:
            if m["display_name"] == name:
                ordered.append(m)
                break
    # Add any remaining models not in kl_order
    for m in kl_models:
        if m not in ordered:
            ordered.append(m)

    if not ordered:
        print("  [!] No models with KL data for chart, skipping.")
        return None

    names = [m["display_name"] for m in ordered]
    kl_means = [m["kl_mean"] for m in ordered]
    kl_999s = [m.get("kl_99_9", 0) or 0 for m in ordered]

    n = len(names)
    x = np.arange(n)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars_mean = ax.bar(x - bar_width / 2, kl_means, bar_width,
                       color="#1565C0", edgecolor="white", linewidth=0.5,
                       label="KL Mean")
    bars_999 = ax.bar(x + bar_width / 2, kl_999s, bar_width,
                      color="#E65100", edgecolor="white", linewidth=0.5,
                      label="KL 99.9th Percentile")

    # Value labels
    for bar_group, vals in [(bars_mean, kl_means), (bars_999, kl_999s)]:
        for bar, val in zip(bar_group, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, val * 1.15,
                        f"{val:.4f}" if val < 0.1 else f"{val:.3f}",
                        ha="center", va="bottom", fontsize=8, rotation=45)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=35, ha="right")

    # Color the x-tick labels by model category
    for i, m in enumerate(ordered):
        color = MODEL_COLORS.get(m["display_name"], "#333333")
        ax.get_xticklabels()[i].set_color(color)
        ax.get_xticklabels()[i].set_fontweight("bold")

    _apply_style(ax, "KL Divergence from F16 Reference",
                 "", "KL Divergence (log scale, lower is better)")
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
    fig.tight_layout()

    out = os.path.join(output_dir, "kl_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Plot 6: Efficiency bubble chart
# ---------------------------------------------------------------------------

def plot_efficiency(models, output_dir):
    """Bubble chart: X=Size (GB), Y=Speed, bubble size ~ 1/PPL."""
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    legend_cats = set()
    texts = []

    # Compute bubble sizes: scale 1/PPL to a reasonable pixel area
    inv_ppls = [1.0 / m["perplexity"] for m in models]
    min_inv = min(inv_ppls)
    max_inv = max(inv_ppls)
    inv_range = max_inv - min_inv if max_inv != min_inv else 1.0

    for m in models:
        cat = m["category"]
        color = COLORS[cat]
        label = CATEGORY_LABELS[cat] if cat not in legend_cats else None
        legend_cats.add(cat)

        inv_ppl = 1.0 / m["perplexity"]
        # Scale bubble area: min 120, max 600
        bubble_size = 120 + 480 * (inv_ppl - min_inv) / inv_range

        ax.scatter(
            m["size_gb"], m["tg128_ts"],
            s=bubble_size, color=color, alpha=0.7, zorder=5,
            edgecolors="white", linewidths=0.8, label=label,
        )
        texts.append(ax.text(m["size_gb"], m["tg128_ts"],
                             m["display_name"], fontsize=8))

    # Use adjustText to reposition overlapping labels
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    # Add annotation explaining bubble size
    ax.annotate(
        "Bubble size = 1/PPL\n(larger = better quality)",
        xy=(0.02, 0.02), xycoords="axes fraction",
        fontsize=8, color="#757575", ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#BDBDBD", alpha=0.9),
    )

    _apply_style(ax, "Efficiency: Size vs Speed (Quality as Bubble Size)",
                 "Model Size (GB)", "Speed tg128 (tokens/sec)")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    fig.tight_layout()

    out = os.path.join(output_dir, "efficiency.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Plot 7: APEX vs Unsloth KL Divergence
# ---------------------------------------------------------------------------

def plot_kl_apex_vs_unsloth(models, output_dir):
    """Grouped bar chart: APEX (green) vs Unsloth (orange) vs Q8_0 (red).

    Bars show KL Mean, with KL 99.9% as error-bar-like markers on top.
    Only models with actual KL data are included (F16, unsloth_q8_k_xl excluded
    because they have null KL values).
    """
    # Define model order and expected display names
    model_order = [
        "APEX I-Quality (21.3 GB)",
        "APEX I-Balanced (23.6 GB)",
        "APEX I-Compact (16.1 GB)",
        "APEX Mini (12.2 GB)",
        "Unsloth Q8_K_XL (45.3 GB)",
        "Unsloth Q4_K_L (18.8 GB)",
        "Q8_0 (34.4 GB)",
    ]

    # Color assignment per group
    bar_colors = {
        "APEX I-Quality (21.3 GB)":     "#388E3C",
        "APEX I-Balanced (23.6 GB)":    "#388E3C",
        "APEX I-Compact (16.1 GB)":     "#388E3C",
        "APEX Mini (12.2 GB)":          "#388E3C",
        "Unsloth Q8_K_XL (45.3 GB)":   "#F57C00",
        "Unsloth Q4_K_L (18.8 GB)":    "#F57C00",
        "Q8_0 (34.4 GB)":              "#D32F2F",
    }

    # Filter to only models with KL data, in order
    ordered = []
    for name in model_order:
        for m in models:
            if m["display_name"] == name:
                if m.get("kl_mean") is not None and m["kl_mean"] is not None:
                    ordered.append(m)
                break

    if not ordered:
        print("  [!] No models with KL data for APEX vs Unsloth chart, skipping.")
        return None

    names = [m["display_name"] for m in ordered]
    kl_means = [m["kl_mean"] for m in ordered]
    kl_999s = [m.get("kl_99_9", 0) or 0 for m in ordered]
    colors = [bar_colors.get(n, "#999999") for n in names]

    n = len(names)
    x = np.arange(n)
    bar_width = 0.6

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Bars for KL Mean
    bars = ax.bar(x, kl_means, bar_width, color=colors,
                  edgecolor="white", linewidth=0.5, label="KL Mean", zorder=3)

    # KL 99.9% as marker points above the bars
    ax.scatter(x, kl_999s, marker='v', s=80, color='#333333', zorder=5,
               label="KL 99.9th Percentile")
    # Connect bar top to 99.9% marker with thin lines
    for i in range(n):
        ax.plot([x[i], x[i]], [kl_means[i], kl_999s[i]],
                color='#666666', linewidth=1.0, linestyle='--', zorder=4)

    # Value labels on bars
    for i, (bar, mean_val, p999_val) in enumerate(zip(bars, kl_means, kl_999s)):
        ax.text(bar.get_x() + bar.get_width() / 2, mean_val + 0.001,
                f"{mean_val:.4f}", ha="center", va="bottom", fontsize=8,
                fontweight="bold")
        ax.text(x[i] + 0.15, p999_val + 0.02,
                f"{p999_val:.3f}", ha="left", va="bottom", fontsize=7,
                color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=35, ha="right")

    # Color the x-tick labels to match bar colors
    for i, name in enumerate(names):
        ax.get_xticklabels()[i].set_color(colors[i])
        ax.get_xticklabels()[i].set_fontweight("bold")

    # Custom legend with group colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#388E3C", edgecolor="white", label="APEX Configurations"),
        Patch(facecolor="#F57C00", edgecolor="white", label="Unsloth"),
        Patch(facecolor="#D32F2F", edgecolor="white", label="Q8_0 Baseline"),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='#333333',
                   markersize=8, label="KL 99.9th Percentile"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper left",
              framealpha=0.9)

    _apply_style(ax, "APEX vs Unsloth: KL Divergence from F16",
                 "", "KL Divergence (lower is better)")
    fig.tight_layout()

    out = os.path.join(output_dir, "kl_apex_vs_unsloth.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality benchmark comparison plots "
                    "from JSON files."
    )
    parser.add_argument(
        "--input-dir",
        default="benchmark_results/final/",
        help="Directory containing benchmark JSON files "
             "(default: benchmark_results/final/).",
    )
    parser.add_argument(
        "--output-dir",
        default="plots/",
        help="Directory to save generated PNG plots (default: plots/).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    models = load_benchmarks(args.input_dir)

    plots = [
        ("1/7", "Pareto: Perplexity vs Model Size",
         plot_pareto_ppl_size),
        ("2/7", "Pareto: Perplexity vs Speed",
         plot_pareto_ppl_speed),
        ("3/7", "Radar chart",
         plot_radar_chart),
        ("4/7", "Accuracy benchmark comparison",
         plot_accuracy_comparison),
        ("5/7", "KL Divergence comparison",
         plot_kl_comparison),
        ("6/7", "Efficiency bubble chart",
         plot_efficiency),
        ("7/7", "APEX vs Unsloth KL Divergence",
         plot_kl_apex_vs_unsloth),
    ]

    generated = []
    for step, desc, fn in plots:
        print(f"\n[{step}] Generating {desc} ...")
        result = fn(models, args.output_dir)
        if result:
            generated.append(result)
            print(f"       Saved: {result}")

    print(f"\nDone. {len(generated)} plot(s) generated in {args.output_dir}")
    for path in generated:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
