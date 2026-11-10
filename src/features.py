"""Turn every labeled coin into a feature vector.

Features per coin:
  r                      radius in pixels (camera height was kept constant)
  h_in, s_in, v_in       mean HSV colour of the inner disk  (0 .. 0.5 r)
  h_out, s_out, v_out    mean HSV colour of the outer ring  (0.70 .. 0.90 r)

Bi-metallic coins (50 kr, 1 TL, 5 TL) have different inner and outer colours,
so the two regions are measured separately. The ring stops at 0.90 r to stay
away from the coin edge and the paper.

Output: data/features.csv
"""
import csv
import os

import cv2
import numpy as np

from detect import SCALE

LABELS_CSV = "data/labels.csv"
COINS_CSV = "data/coins.csv"
RAW_DIR = "data/raw"
OUT_CSV = "data/features.csv"
CLASSES = ["5kr", "10kr", "25kr", "50kr", "1tl", "5tl"]

INNER = (0.0, 0.50)   # inner disk, as a fraction of the radius
OUTER = (0.70, 0.90)  # outer ring


def region_mask(shape, x, y, r, r_from, r_to):
    """Binary mask of the ring between r_from*r and r_to*r (a disk if r_from == 0)."""
    mask = np.zeros(shape[:2], np.uint8)
    cv2.circle(mask, (x, y), int(r * r_to), 255, -1)
    if r_from > 0:
        cv2.circle(mask, (x, y), int(r * r_from), 0, -1)
    return mask


def region_mean(hsv, mask):
    """Mean H, S, V inside the mask."""
    # TODO (you): compute the mean H, S and V of the pixels where mask is 255.
    # Hint: cv2.mean(image, mask=...) returns a tuple of 4 values (the 4th is unused).
    pass


def load_groups():
    """Images that are the same layout (e.g. a coin set flipped over) share a group,
    so they always end up in the same train/test split."""
    groups = {}
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            note = (row.get("note") or "").strip()
            groups[row["file"]] = note if "pair" in note else row["file"]
    return groups


def main():
    groups = load_groups()

    coins_by_file = {}
    with open(COINS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["label"] in CLASSES:            # skip not_coin and unsure
                coins_by_file.setdefault(row["file"], []).append(row)

    rows = []
    for name, coins in coins_by_file.items():
        img = cv2.imread(os.path.join(RAW_DIR, name))
        if img is None:
            print("Could not read", name)
            continue
        img = cv2.resize(img, None, fx=SCALE, fy=SCALE)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        for c in coins:
            x, y, r = int(c["x"]), int(c["y"]), int(c["r"])
            inner = region_mean(hsv, region_mask(hsv.shape, x, y, r, *INNER))
            outer = region_mean(hsv, region_mask(hsv.shape, x, y, r, *OUTER))
            rows.append([name, c["coin_id"], c["label"], groups.get(name, name), r,
                         *[round(v, 2) for v in inner], *[round(v, 2) for v in outer]])

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "coin_id", "label", "group", "r",
                         "h_in", "s_in", "v_in", "h_out", "s_out", "v_out"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} coins to {OUT_CSV}")


if __name__ == "__main__":
    main()
