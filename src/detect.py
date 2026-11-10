import cv2
import numpy as np

SCALE = 0.25


def detect_coins(img):
    """Return a list of (x, y, r) circles found in a BGR image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=60,
        param1=100,
        param2=30,
        minRadius=20,
        maxRadius=70,
    )

    if circles is None:
        return []
    circles = np.round(circles[0]).astype(int)
    return [(x, y, r) for x, y, r in circles]


def main():
    files = ["trial_01.jpg", "trial_02.jpg", "trial_03.jpg"]

    for name in files:
        img = cv2.imread("data/raw/" + name)
        if img is None:
            print("Could not read", name)
            continue

        img = cv2.resize(img, None, fx=SCALE, fy=SCALE)
        coins = detect_coins(img)
        print(name + ": Found", len(coins), "coins")

        for x, y, r in coins:
            print(f"   x={x} y={y} r={r}")
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)   # outline
            cv2.circle(img, (x, y), 2, (0, 0, 255), 3)   # center

        cv2.imwrite("outputs/det_" + name, img)


if __name__ == "__main__":
    main()