import cv2

# 1. Read the image
img = cv2.imread("data/raw/trial_01.jpg")

# imread does not raise an error if the file is missing; it returns None.
if img is None:
    print("Image not found! Check the file path.")
    exit()

# 2. Print the original size: (height, width, channels)
print("Original size:", img.shape)

# 3. Downscale the image to 1/4 of its size
scale = 0.25
small = cv2.resize(img, None, fx=scale, fy=scale)
print("Resized size:", small.shape)

# 4. Save the small version
cv2.imwrite("outputs/trial_01_small.jpg", small)
print("Saved: outputs/trial_01_small.jpg")