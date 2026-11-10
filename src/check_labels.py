"""Compare the per-coin labels (coins.csv) with the hand counts (labels.csv).

Prints, for every image:
  - how many coins were in the photo (hand count)
  - how many real coins Hough found, and how many false detections
  - per-class differences between the two labelings

and a detection summary (precision / recall) for the whole dataset.
"""
import csv
from collections import Counter

LABELS_CSV = "data/labels.csv"
COINS_CSV = "data/coins.csv"
CLASSES = ["5kr", "10kr", "25kr", "50kr", "1tl", "5tl"]


def load_truth():
    truth = {}
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            counts = {c: int(row[c]) for c in CLASSES}
            truth[row["file"]] = (counts, row.get("note", "") or "")
    return truth


def load_coin_labels():
    per_file = {}
    with open(COINS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            per_file.setdefault(row["file"], Counter())[row["label"]] += 1
    return per_file


def main():
    truth = load_truth()
    labeled = load_coin_labels()

    total_true = total_found = total_false = total_unsure = 0
    class_totals = Counter()

    print(f"{'file':<14}{'true':>5}{'found':>6}{'false':>6}{'unsure':>7}  class differences (tool - hand)")
    for name, (counts, note) in truth.items():
        got = labeled.get(name, Counter())
        n_true = sum(counts.values())
        n_false = got["not_coin"]
        n_unsure = got["unsure"]
        n_found = sum(got[c] for c in CLASSES) + n_unsure   # real coins Hough found

        total_true += n_true
        total_found += n_found
        total_false += n_false
        total_unsure += n_unsure
        for c in CLASSES:
            class_totals[c] += got[c]

        diffs = [f"{c}:{got[c] - counts[c]:+d}" for c in CLASSES if got[c] != counts[c]]
        flag = ""
        if name not in labeled:
            flag = "  NOT LABELED"
        elif n_found != n_true:
            flag = "  <-- coin count mismatch"
        elif diffs and n_unsure == 0:
            flag = "  <-- check labels"
        if "uncertain" in note:
            flag += "  (5/10 uncertain)"
        print(f"{name:<14}{n_true:>5}{n_found:>6}{n_false:>6}{n_unsure:>7}  {' '.join(diffs)}{flag}")

    print()
    print("Detection summary")
    print(f"  coins in photos (hand count): {total_true}")
    print(f"  real coins found by Hough   : {total_found}")
    print(f"  false detections            : {total_false}")
    if total_found + total_false:
        print(f"  precision: {total_found / (total_found + total_false):.3f}")
    if total_true:
        print(f"  recall   : {total_found / total_true:.3f}")
    print()
    print("Labeled coins per class:", dict(class_totals), "| unsure:", total_unsure)


if __name__ == "__main__":
    main()