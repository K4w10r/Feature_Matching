# Feature Matching with Homography between two images
import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

#%%
# === Parameters ===
MIN_MATCH_COUNT = 10  # Minimum good matches required for homography

# === Load images ===
img1 = cv.imread("photo_2_query.jpg", cv.IMREAD_GRAYSCALE)   # Query Image
img2 = cv.imread("photo_2_train.jpg", cv.IMREAD_GRAYSCALE)   # Train Image

# === Detect SIFT keypoints and descriptors ===
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# === FLANN-based Matcher setup ===
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# === Lowe's ratio test ===
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

#%%
# === Compute Homography if enough matches ===
if len(good_matches) >= MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    # Draw detected region in train image (transformed query corners)
    h, w = img1.shape
    corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(corners, H)
    img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

else:
    print(f"Not enough matches are found - {len(good_matches)}/{MIN_MATCH_COUNT}")
    matchesMask = None

# === Draw matches ===
draw_params = dict(matchColor=(0, 255, 0),  # Green matches
                   singlePointColor=None,
                   matchesMask=matchesMask,  # Only inliers if homography successful
                   flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

result_img = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)

# === Show and Save result ===

cv.imshow("Matches", result_img)
cv.waitKey()
cv.destroyAllWindows()
"""plt.figure(figsize=(15, 8))
plt.title("Feature Matching with Homography")
plt.imshow(result_img, cmap='gray')
plt.axis('off')
plt.show()"""

cv.imwrite("02_Feature_Matching_Homography.jpg", result_img)
