"""Create the figures used in the README and the portfolio.

Outputs (in figures/):
  feature_space.png   every coin in the (radius, saturation difference) space
  radius_by_class.png radius distribution per class: why size alone is not enough
  coin_grid.png       sample crops of every class from the dataset
"""
import glob
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FEATURES_CSV = "data/features.csv"
CROP_DIR = "data/crops"
OUT_DIR = "figures"

CLASSES = ["5kr", "10kr", "25kr", "50kr", "1tl", "5tl"]
SHORT = {"5kr": "5 kr", "10kr": "10 kr", "25kr": "25 kr",
         "50kr": "50 kr", "1tl": "1 TL", "5tl": "5 TL"}   # ASCII: cv2.putText has no 'ş'
NAMES = {"5kr": "5 kuruş", "10kr": "10 kuruş", "25kr": "25 kuruş",
         "50kr": "50 kuruş", "1tl": "1 lira", "5tl": "5 lira"}
# Categorical palette in fixed order (validated for colour-vision deficiency).
# Each class also gets its own marker shape, so colour is never the only cue.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]

INK = "#2b2b2b"
MUTED = "#6b6b6b"
GRID = "#e6e6e3"


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def feature_space(df):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for c, color, marker in zip(CLASSES, COLORS, MARKERS):
        d = df[df["label"] == c]
        ax.scatter(d["r"], d["s_diff"], s=42, c=color, marker=marker,
                   edgecolors="white", linewidths=1.2, label=NAMES[c], alpha=0.9)
        ax.annotate(NAMES[c], (d["r"].median(), d["s_diff"].median()),
                    xytext=(0, 14), textcoords="offset points", ha="center",
                    fontsize=9, color=INK, fontweight="bold")
    style(ax)
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Radius (px, image downscaled to 25%)", color=INK)
    ax.set_ylabel("Saturation: inner disk − outer ring", color=INK)
    ax.set_title("Each coin as two numbers: size and bi-metal contrast",
                 color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "feature_space.png"), dpi=200)
    plt.close(fig)


def radius_by_class(df):
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for i, (c, color, marker) in enumerate(zip(CLASSES, COLORS, MARKERS)):
        r = df.loc[df["label"] == c, "r"].values
        y = np.full(len(r), i) + rng.uniform(-0.18, 0.18, len(r))
        ax.scatter(r, y, s=26, c=color, marker=marker, edgecolors="white",
                   linewidths=0.8, alpha=0.9)
        ax.plot([np.median(r)] * 2, [i - 0.3, i + 0.3], color=INK, linewidth=2)
    style(ax)
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels([NAMES[c] for c in CLASSES], color=INK)
    ax.set_xlabel("Radius (px, image downscaled to 25%)  ·  black tick = median", color=INK)
    ax.set_title("Size alone overlaps for neighbouring coins (5 vs 10 kuruş, 1 vs 5 lira)",
                 color=INK, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "radius_by_class.png"), dpi=200)
    plt.close(fig)


def coin_grid(per_class=6, size=140):
    rows = []
    for c in CLASSES:
        files = sorted(glob.glob(os.path.join(CROP_DIR, c, "*.jpg")))
        step = max(1, len(files) // per_class)
        tiles = []
        for path in files[::step][:per_class]:
            img = cv2.imread(path)
            tiles.append(cv2.resize(img, (size, size)) if img is not None
                         else np.full((size, size, 3), 255, np.uint8))
        while len(tiles) < per_class:
            tiles.append(np.full((size, size, 3), 255, np.uint8))
        label = np.full((size, 150, 3), 255, np.uint8)
        cv2.putText(label, SHORT[c], (8, size // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (43, 43, 43), 2)
        rows.append(np.hstack([label] + tiles))
    grid = np.vstack(rows)
    cv2.imwrite(os.path.join(OUT_DIR, "coin_grid.png"), grid)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(FEATURES_CSV)
    df["s_diff"] = df["s_in"] - df["s_out"]
    feature_space(df)
    radius_by_class(df)
    coin_grid()
    print("Saved figures/feature_space.png, figures/radius_by_class.png, figures/coin_grid.png")


if __name__ == "__main__":
    main()
