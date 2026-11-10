"""Draw every labeled coin on its photo, numbered with its coin_id and label.

Usage:
    python src/review.py                     # all images with a mismatch
    python src/review.py IMG_8491.jpg ...    # only the given images

Output: outputs/review/<image>.jpg
Open the image, find the coin whose label is wrong, and fix that row
(file + coin_id) in data/coins.csv, or fix the count in data/labels.csv.
"""
import csv
import os
import sys
from collections import Counter

import cv2

from detect import SCALE

LABELS_CSV = "data/labels.csv"
COINS_CSV = "data/coins.csv"
RAW_DIR = "data/raw"
OUT_DIR = "outputs/review"
CLASSES = ["5kr", "10kr", "25kr", "50kr", "1tl", "5tl"]

COLORS = {
    "5kr": (0, 165, 255),
    "10kr": (0, 255, 255),
    "25kr": (255, 255, 0),
    "50kr": (255, 0, 255),
    "1tl": (0, 255, 0),
    "5tl": (255, 0, 0),
    "not_coin": (0, 0, 0),
    "unsure": (0, 0, 255),
}


def mismatched_files():
    truth = {}
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            truth[row["file"]] = {c: int(row[c]) for c in CLASSES}
    got = {}
    with open(COINS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            got.setdefault(row["file"], Counter())[row["label"]] += 1
    return [name for name, counts in truth.items()
            if any(got.get(name, Counter())[c] != counts[c] for c in CLASSES)]


def main():
    files = sys.argv[1:] or mismatched_files()

    coins = {}
    with open(COINS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            coins.setdefault(row["file"], []).append(row)

    os.makedirs(OUT_DIR, exist_ok=True)
    for name in files:
        img = cv2.imread(os.path.join(RAW_DIR, name))
        if img is None:
            print("Could not read", name)
            continue
        img = cv2.resize(img, None, fx=SCALE * 2, fy=SCALE * 2)   # half resolution

        for row in coins.get(name, []):
            x, y, r = (int(int(row[k]) * 2) for k in ("x", "y", "r"))
            color = COLORS.get(row["label"], (128, 128, 128))
            cv2.circle(img, (x, y), r, color, 4)
            text = f"{row['coin_id']}:{row['label']}"
            cv2.putText(img, text, (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 0), 5)
            cv2.putText(img, text, (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 2)

        out = os.path.join(OUT_DIR, name)
        cv2.imwrite(out, img)
        print("Saved", out)


if __name__ == "__main__":
    main()