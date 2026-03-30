#!/usr/bin/env python3
"""Generate benchmark chart images for the tokie README."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
BG = "#1a1a2e"
FG = "#e0e0e0"
GRID = "#2a2a4a"
TOKIE_COLOR = "#ff6b35"
HF_COLOR = "#4a9eff"
TIKTOKEN_COLOR = "#4a9eff"

SUBTITLE = "Apple M3 Pro  \u2022  enwik8"
DPI = 200
ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(ASSETS, exist_ok=True)


def _style(ax, title, subtitle=SUBTITLE):
    ax.set_facecolor(BG)
    ax.figure.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=FG, labelsize=10)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.grid(axis="x", color=GRID, linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_title(title, color=FG, fontsize=14, fontweight="bold", pad=18)
    if subtitle:
        ax.text(
            0.5, 1.02, subtitle,
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, color="#888888",
        )


def _save(fig, name):
    for ext in ("png", "svg"):
        path = os.path.join(ASSETS, f"{name}.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  saved {path}")
    plt.close(fig)


def _horizontal_bars(labels, tokie_vals, other_vals, other_label, title, fname,
                     xlabel="Time (ms)", speedups=None, figsize=(9, None)):
    n = len(labels)
    h = max(3.0, n * 0.72 + 1.2) if figsize[1] is None else figsize[1]
    fig, ax = plt.subplots(figsize=(figsize[0], h))
    _style(ax, title)

    bar_h = 0.35
    y_pos = range(n)

    bars_other = ax.barh(
        [y - bar_h / 2 for y in y_pos], other_vals,
        height=bar_h, color=HF_COLOR, label=other_label, zorder=3,
    )
    bars_tokie = ax.barh(
        [y + bar_h / 2 for y in y_pos], tokie_vals,
        height=bar_h, color=TOKIE_COLOR, label="tokie", zorder=3,
    )

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()

    # Speedup annotations
    if speedups:
        for i, sp in enumerate(speedups):
            x = other_vals[i]
            ax.text(
                x + max(other_vals) * 0.02, i - bar_h / 2,
                f"{sp}",
                va="center", ha="left", color=FG, fontsize=9, fontweight="bold",
            )

    ax.legend(loc="lower right", facecolor="#22223a", edgecolor=GRID,
              labelcolor=FG, fontsize=9)
    _save(fig, fname)


# ---------------------------------------------------------------------------
# 1. Overview benchmark
# ---------------------------------------------------------------------------
def chart_overview():
    labels =   ["BERT (900KB)", "GPT-2 (900KB)", "Llama 3 (900KB)", "Qwen 3 (900KB)"]
    tokie =    [9.84, 9.42, 9.45, 9.58]
    hf =       [280.6, 209.2, 210.8, 230.1]
    speedups = ["29x faster", "22x faster", "22x faster", "24x faster"]
    _horizontal_bars(labels, tokie, hf, "HuggingFace", "Encoding Speed: tokie vs HuggingFace",
                     "benchmark", speedups=speedups)


# ---------------------------------------------------------------------------
# 2. BPE models
# ---------------------------------------------------------------------------
def chart_bpe():
    labels =   ["GPT-2 (900KB)", "Llama 3 (900KB)", "Qwen 3 (900KB)", "ModernBERT (900KB)"]
    tokie =    [9.42, 9.45, 9.58, 9.75]
    hf =       [209.2, 210.8, 230.1, 236.3]
    speedups = ["22x faster", "22x faster", "24x faster", "24x faster"]
    _horizontal_bars(labels, tokie, hf, "HuggingFace", "BPE Models: tokie vs HuggingFace",
                     "benchmark_bpe", speedups=speedups)


# ---------------------------------------------------------------------------
# 3. WordPiece
# ---------------------------------------------------------------------------
def chart_wordpiece():
    labels =   ["BERT (45KB)", "BERT (900KB)"]
    tokie =    [0.56, 9.84]
    hf =       [10.9, 280.6]
    speedups = ["20x faster", "29x faster"]
    _horizontal_bars(labels, tokie, hf, "HuggingFace", "WordPiece: tokie vs HuggingFace",
                     "benchmark_wordpiece", speedups=speedups)


# ---------------------------------------------------------------------------
# 4. SentencePiece
# ---------------------------------------------------------------------------
def chart_sentencepiece():
    labels =   ["Gemma 3 (45KB)", "Gemma 3 (900KB)"]
    tokie =    [5.20, 131.07]
    hf =       [11.6, 329.5]
    speedups = ["2x faster", "3x faster"]
    _horizontal_bars(labels, tokie, hf, "HuggingFace", "SentencePiece: tokie vs HuggingFace",
                     "benchmark_sentencepiece", speedups=speedups)


# ---------------------------------------------------------------------------
# 5. tiktoken
# ---------------------------------------------------------------------------
def chart_tiktoken():
    labels =   ["cl100k (45KB)", "cl100k (900KB)", "o200k (45KB)", "o200k (900KB)"]
    tokie =    [0.69, 9.63, 0.52, 9.83]
    tt =       [2.37, 45.67, 4.10, 81.47]
    speedups = ["3.5x faster", "4.7x faster", "7.9x faster", "8.3x faster"]
    _horizontal_bars(labels, tokie, tt, "tiktoken", "Encoding Speed: tokie vs tiktoken",
                     "benchmark_tiktoken", speedups=speedups)


# ---------------------------------------------------------------------------
# 6. Loading times
# ---------------------------------------------------------------------------
def chart_loading():
    labels =   ["BERT", "GPT-2", "Llama 3.2", "cl100k", "o200k"]
    tokie =    [12.4, 24.9, 80.2, 54.6, 117.6]
    hf =       [100.8, 135.0, 257.6, 173.2, 283.0]
    speedups = ["8.2x faster", "5.4x faster", "3.2x faster", "3.2x faster", "2.4x faster"]
    _horizontal_bars(labels, tokie, hf, "HuggingFace / tiktoken",
                     "Tokenizer Loading Time",
                     "benchmark_loading", xlabel="Time (ms)", speedups=speedups)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating benchmark charts...")
    chart_overview()
    chart_bpe()
    chart_wordpiece()
    chart_sentencepiece()
    chart_tiktoken()
    chart_loading()
    print("Done.")
