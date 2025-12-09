import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#%%

# === Load and preprocess image ===
img = cv.imread('photo_1.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_float = np.float32(gray)

#%%

# === Harris Corner Detection (Top 4 corners) ===
dst = cv.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
dst = cv.dilate(dst, None)

# Find coordinates of top 4 corners

corners = cv.goodFeaturesToTrack(gray, maxCorners=4, qualityLevel=0.01, minDistance=100, useHarrisDetector=True, k=0.04)
#corners = np.int0(corners)
corners = corners.astype(int)


harris_img = img.copy()

for i in corners:
    x, y = i.ravel()
    cv.circle(harris_img, (x, y), 10, 255, -1)


# Save Harris result
cv.imshow('Harris', harris_img)
cv.waitKey()
cv.destroyAllWindows()
cv.imwrite("01_Harris_Top4.jpg", harris_img)

#%%

# === SIFT Keypoint Detection ===
sift = cv.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)

# Draw SIFT keypoints
sift_img = cv.drawKeypoints(gray, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow('SIFT', sift_img)
cv.waitKey()
cv.destroyAllWindows()
cv.imwrite("02_SIFT_Keypoints.jpg", sift_img)

