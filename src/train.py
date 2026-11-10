"""Train and evaluate coin classifiers with photo-grouped cross-validation.

Every coin is predicted by a model that never saw ANY coin from the same photo
(or from the same photo pair), using GroupKFold on the 'group' column.

Two feature sets are compared:
  size only    : r
  size + colour: r + inner/outer HSV means

and two models: k-nearest neighbours and an RBF support vector machine.

Outputs (in outputs/):
  results.csv            accuracy of every model / feature set
  confusion_matrix.png   confusion matrix of the best model
  misclassified.csv      every wrong prediction, for error analysis
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURES_CSV = "data/features.csv"
LABELS_CSV = "data/labels.csv"
OUT_DIR = "outputs"

CLASSES = ["5kr", "10kr", "25kr", "50kr", "1tl", "5tl"]
VALUES = {"5kr": 0.05, "10kr": 0.10, "25kr": 0.25, "50kr": 0.50, "1tl": 1.0, "5tl": 5.0}

FEATURE_SETS = {
    "size only": ["r"],
    "size + colour": ["r", "h_in", "s_in", "v_in", "h_out", "s_out", "v_out"],
    "size + saturation": ["r", "s_in", "s_out"],
    "size + s_diff": ["r", "s_diff"],
}


def make_models():
    return {
        "kNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")),
    }


def exact_total_rate(df, pred):
    """Share of photos whose total value is predicted exactly.
    Only photos without unsure coins and without a '5/10 uncertain' note are used."""
    labels = pd.read_csv(LABELS_CSV)
    note = labels["note"].fillna("")
    hand_total = {row["file"]: sum(int(row[c]) for c in CLASSES) for _, row in labels.iterrows()}
    usable = set(labels.loc[~note.str.contains("uncertain"), "file"])

    df = df.assign(pred=pred)
    ok = n = 0
    for name, g in df.groupby("file"):
        if name not in usable or len(g) != hand_total[name]:   # some coins were 'unsure'
            continue
        true_value = sum(VALUES[c] for c in g["label"])
        pred_value = sum(VALUES[c] for c in g["pred"])
        n += 1
        ok += abs(true_value - pred_value) < 1e-9
    return ok, n


def main():
    df = pd.read_csv(FEATURES_CSV)
    df["s_diff"] = df["s_in"] - df["s_out"]
    y = df["label"].values
    groups = df["group"].values
    cv = GroupKFold(n_splits=5)

    print(f"{len(df)} coins, {df['group'].nunique()} photo groups, 5-fold grouped CV\n")

    results = []
    best = None
    for set_name, cols in FEATURE_SETS.items():
        X = df[cols].values
        for model_name, model in make_models().items():
            pred = cross_val_predict(model, X, y, groups=groups, cv=cv)
            acc = accuracy_score(y, pred)
            ok, n = exact_total_rate(df, pred)
            results.append([set_name, model_name, round(acc, 3), f"{ok}/{n}"])
            print(f"{set_name:<14} {model_name:<10} accuracy = {acc:.3f}   exact photo total: {ok}/{n}")
            if best is None or acc > best[0]:
                best = (acc, set_name, model_name, pred)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "results.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["features", "model", "accuracy", "exact_photo_total"])
        writer.writerows(results)

    acc, set_name, model_name, pred = best
    print(f"\nBest: {model_name} with {set_name} (accuracy {acc:.3f})\n")
    print(classification_report(y, pred, labels=CLASSES, digits=3))

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y, pred, labels=CLASSES, cmap="Blues",
                                            colorbar=False, ax=ax)
    ax.set_title(f"{model_name}, {set_name} — accuracy {acc:.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=200)

    wrong = df.loc[pred != y, ["file", "coin_id", "label", "r"]].assign(pred=pred[pred != y])
    wrong.to_csv(os.path.join(OUT_DIR, "misclassified.csv"), index=False)
    print(f"{len(wrong)} misclassified coins saved to outputs/misclassified.csv")


if __name__ == "__main__":
    main()
