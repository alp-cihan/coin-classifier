"""Interactive tool to label every detected coin with its class.

For each image in data/labels.csv, coins are detected with Hough circles.
Each coin is shown large (cropped from the full-resolution photo) next to an
overview of the whole image. Press a key to assign its class.

The detected radius is deliberately NOT shown: labels must come from looking
at the coin, not from the feature the classifier will later use.

Keys: 1=5kr 2=10kr 3=25kr 4=50kr 5=1tl 6=5tl  x=not a coin  u=unsure  q=quit

Results are appended to data/coins.csv after each finished image, so you can
quit and resume later: images already in coins.csv are skipped.
"""
import csv
import os

import cv2
import numpy as np

from detect import detect_coins, SCALE

LABELS_CSV = "data/labels.csv"
COINS_CSV = "data/coins.csv"
RAW_DIR = "data/raw"
CROP_DIR = "data/crops"

KEYS = {
    "1": "5kr",
    "2": "10kr",
    "3": "25kr",
    "4": "50kr",
    "5": "1tl",
    "6": "5tl",
    "x": "not_coin",
    "u": "unsure",
}

VIEW_HEIGHT = 600   # height of the overview image in the window
CROP_SIZE = 400     # size of the enlarged coin crop in the window


def read_file_list():
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        return [row["file"] for row in csv.DictReader(f)]


def already_labeled():
    if not os.path.exists(COINS_CSV):
        return set()
    with open(COINS_CSV, newline="", encoding="utf-8") as f:
        return {row["file"] for row in csv.DictReader(f)}


def crop_coin(img, x, y, r, margin=1.3):
    """Square crop around a coin, a bit larger than the coin itself."""
    h, w = img.shape[:2]
    half = int(r * margin)
    x0, x1 = max(0, x - half), min(w, x + half)
    y0, y1 = max(0, y - half), min(h, y + half)
    return img[y0:y1, x0:x1]


def make_view(full, x, y, r, index, total, name):
    """Window content: overview on the left, enlarged coin on the right."""
    # Coordinates come from the downscaled image; convert to full resolution.
    fx, fy, fr = int(x / SCALE), int(y / SCALE), int(r / SCALE)

    overview = full.copy()
    cv2.circle(overview, (fx, fy), fr, (0, 0, 255), 12)
    k = VIEW_HEIGHT / overview.shape[0]
    overview = cv2.resize(overview, None, fx=k, fy=k)

    crop = cv2.resize(crop_coin(full, fx, fy, fr), (CROP_SIZE, CROP_SIZE))

    ow = overview.shape[1]
    canvas = np.full((VIEW_HEIGHT, ow + CROP_SIZE, 3), 40, np.uint8)
    canvas[:, :ow] = overview
    canvas[:CROP_SIZE, ow:] = crop

    lines = [
        f"{name}",
        f"coin {index}/{total}",
        "1=5kr  2=10kr  3=25kr",
        "4=50kr 5=1tl   6=5tl",
        "x=not coin  u=unsure",
        "q=quit",
    ]
    for i, text in enumerate(lines):
        cv2.putText(canvas, text, (ow + 10, CROP_SIZE + 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return canvas


def save_crop(full, x, y, r, label, name, coin_id):
    folder = os.path.join(CROP_DIR, label)
    os.makedirs(folder, exist_ok=True)
    fx, fy, fr = int(x / SCALE), int(y / SCALE), int(r / SCALE)
    stem = os.path.splitext(name)[0]
    path = os.path.join(folder, f"{stem}_{coin_id}.jpg")
    cv2.imwrite(path, crop_coin(full, fx, fy, fr, margin=1.05))


def main():
    files = read_file_list()
    done = already_labeled()

    is_new = not os.path.exists(COINS_CSV)
    out = open(COINS_CSV, "a", newline="", encoding="utf-8")
    writer = csv.writer(out)
    if is_new:
        writer.writerow(["file", "coin_id", "x", "y", "r", "label"])

    for name in files:
        if name in done:
            print("Skip (already labeled):", name)
            continue

        full = cv2.imread(os.path.join(RAW_DIR, name))
        if full is None:
            print("Could not read", name)
            continue

        small = cv2.resize(full, None, fx=SCALE, fy=SCALE)
        coins = detect_coins(small)

        rows = []
        for i, (x, y, r) in enumerate(coins):
            cv2.imshow("label tool", make_view(full, x, y, r, i + 1, len(coins), name))

            while True:
                key = chr(cv2.waitKey(0) & 0xFF).lower()
                if key == "q":
                    out.close()
                    cv2.destroyAllWindows()
                    print("Stopped. Finished images are saved; this one will be shown again.")
                    return
                if key in KEYS:
                    break

            label = KEYS[key]
            rows.append([name, i, x, y, r, label])
            save_crop(full, x, y, r, label, name, i)

        # Write the whole image at once, so an image is never half-saved.
        writer.writerows(rows)
        out.flush()
        print(f"{name}: {len(rows)} coins labeled")

    out.close()
    cv2.destroyAllWindows()
    print("All images labeled. Results in", COINS_CSV)


if __name__ == "__main__":
    main()