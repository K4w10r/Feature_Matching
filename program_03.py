import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')

# === Parameters ===
MIN_MATCH_COUNT = 10  # Minimum number of good matches to compute homography

# === Load the training image (object template) ===
train_img = cv.imread("photo_3_train.jpg", cv.IMREAD_GRAYSCALE)
if train_img is None:
    raise FileNotFoundError("photo_3_train.jpg not found.")

# === Initialize SIFT detector and compute descriptors ===
sift = cv.SIFT_create()
kp_train, des_train = sift.detectAndCompute(train_img, None)

# === Setup FLANN matcher ===
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)

# === Load the video to analyze ===
cap = cv.VideoCapture("video_3_query.mp4")
if not cap.isOpened():
    raise IOError("Cannot open video_3_query.mp4")

# === Optional: Get dimensions to write output video ===
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS)

# Optional output writer
# out = cv.VideoWriter('output_tracking.avi', cv.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # === Detect keypoints and descriptors in current frame ===
    kp_frame, des_frame = sift.detectAndCompute(frame_gray, None)
    if des_frame is None or len(kp_frame) < 2:
        continue

    # === Match features ===
    matches = flann.knnMatch(des_train, des_frame, k=2)

    # === Lowe's ratio test ===
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) >= MIN_MATCH_COUNT:
        src_pts = np.float32([kp_train[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        if H is not None:
            h, w = train_img.shape
            corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            projected_corners = cv.perspectiveTransform(corners, H)

            # === Draw rectangle around detected object ===
            cv.polylines(frame, [np.int32(projected_corners)], True, (0, 255, 0), 3, cv.LINE_AA)

    # === Show the result ===
    cv.imshow("Object Tracking (Train Image Detected in Video)", frame)
    # out.write(frame)  # Optional save to video

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
# out.release()
cv.destroyAllWindows()
