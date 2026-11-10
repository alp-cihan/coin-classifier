"""Detect, classify and count the coins in one photo, and draw the result.

Usage:
    python src/predict.py data/raw/IMG_8493.jpg

The classifier (SVM, size + s_diff) is trained on every labeled photo EXCEPT
the one being predicted (and its pair), so the demo shows an honest,
unseen-photo result rather than a memorised one.

Output: outputs/pred_<image>.jpg
"""
import os
import sys

import cv2
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from detect import SCALE, detect_coins
from features import INNER, OUTER, region_mask, region_mean

FEATURES_CSV = "data/features.csv"
COINS_CSV = "data/coins.csv"
OUT_DIR = "outputs"
FEATURES = ["r", "s_diff"]
VALUES = {"5kr": 0.05, "10kr": 0.10, "25kr": 0.25, "50kr": 0.50, "1tl": 1.0, "5tl": 5.0}
COLORS = {
    "5kr": (0, 165, 255), "10kr": (0, 255, 255), "25kr": (255, 255, 0),
    "50kr": (255, 0, 255), "1tl": (0, 200, 0), "5tl": (255, 0, 0),
}


def train_without(name):
    df = pd.read_csv(FEATURES_CSV)
    df["s_diff"] = df["s_in"] - df["s_out"]
    held_out = df.loc[df["file"] == name, "group"].unique()
    train = df[~df["group"].isin(held_out)]
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale"))
    model.fit(train[FEATURES].values, train["label"].values)
    return model, len(train)


def coin_features(hsv, x, y, r):
    s_in = region_mean(hsv, region_mask(hsv.shape, x, y, r, *INNER))[1]
    s_out = region_mean(hsv, region_mask(hsv.shape, x, y, r, *OUTER))[1]
    return [r, s_in - s_out]


def true_labels(name):
    """Hand labels for this photo, if it was labeled: {(x, y): label}."""
    if not os.path.exists(COINS_CSV):
        return {}
    coins = pd.read_csv(COINS_CSV)
    coins = coins[coins["file"] == name]
    return {(int(r.x), int(r.y)): r.label for r in coins.itertuples()}


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py <image>")
        return
    path = sys.argv[1]
    name = os.path.basename(path)

    img = cv2.imread(path)
    if img is None:
        print("Could not read", path)
        return
    small = cv2.resize(img, None, fx=SCALE, fy=SCALE)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    model, n_train = train_without(name)
    coins = detect_coins(small)
    truth = true_labels(name)

    view = cv2.resize(img, None, fx=SCALE * 2, fy=SCALE * 2)   # half resolution for drawing
    total = 0.0
    true_total = 0.0
    n_wrong = 0
    for x, y, r in coins:
        label = model.predict([coin_features(hsv, x, y, r)])[0]
        total += VALUES[label]

        t = truth.get((x, y))
        wrong = t in VALUES and t != label
        if t in VALUES:
            true_total += VALUES[t]
        n_wrong += wrong

        X, Y, R = 2 * x, 2 * y, 2 * r
        cv2.circle(view, (X, Y), R, (0, 0, 255) if wrong else COLORS[label], 5)
        text = label + (f" ({t})" if wrong else "")
        cv2.putText(view, text, (X - R, Y - R - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6)
        cv2.putText(view, text, (X - R, Y - R - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

    header = f"{len(coins)} coins   predicted total: {total:.2f} TL"
    if truth:
        header += f"   true: {true_total:.2f} TL   errors: {n_wrong}"
    cv2.rectangle(view, (0, 0), (view.shape[1], 70), (40, 40, 40), -1)
    cv2.putText(view, header, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "pred_" + name)
    cv2.imwrite(out, view)
    print(header)
    print(f"Model trained on {n_train} coins from other photos. Saved {out}")


if __name__ == "__main__":
    main()
