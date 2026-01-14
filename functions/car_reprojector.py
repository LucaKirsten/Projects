#Used to match features of image with vehicle to panorama to transpose vehicle location 
# still need to import and compute vehicle 

import numpy as np
import cv2 
import matplotlib.pyplot as plt
import time
import glob 
import os
import json

def display(panorama_with_box, frame_img, good_draw=None, kp1=None, kp2=None):
    """
    Displays feature matches between frame and panorama on one plot,
    where the panorama already includes the reprojected box.

    Args:
        panorama_with_box (ndarray): Panorama image with reprojected car location.
        frame_img (ndarray): Frame image containing the detected vehicle.
        good_draw (list, optional): List of good matches for drawMatchesKnn visualization.
        kp1, kp2 (list, optional): Keypoints for frame and panorama respectively.
    """
    if good_draw is not None and kp1 is not None and kp2 is not None:
        # Use the color panorama (with box) to retain bounding visualization
        match_img = cv2.drawMatchesKnn(
            frame_img, kp1,
            panorama_with_box, kp2,
            good_draw, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        plt.figure(figsize=(14, 6))
        plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
        plt.title("Feature Matches and Reprojected Box on Panorama", fontsize=12)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    else:
        # Fallback: just show the two images side by side if no matches available
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].imshow(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Detected Vehicle Frame", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(cv2.cvtColor(panorama_with_box, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Vehicle Location Reprojected onto Panorama", fontsize=12)
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()



def Knn_matches(des1, des2, k):
    # select k matches for each descriptor 
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1,des2,k) # note k = 2

    # Apply ratio test
    good_draw = [] # need to return list of list to draw function 
    good = [] # list to store matches 

    for m,n in matches:
        if m.distance < 0.9*n.distance: # note it works better with high ratio
            good_draw.append([m])
            good.append(m)
    return good,good_draw

def normal_BF(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches_draw = []
    matches = bf.match(des1,des2)
    sorted_matches = sorted(matches, key= lambda x:x.distance)
    for i in sorted_matches:
        matches_draw.append([i])
    return sorted_matches, matches_draw


def Flann_matcher(des1, des2, k):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # higher = more accurate but slower

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k) # note k = 2 

    # Apply Lowe's ratio test
    good = []
    good_draw = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance: # keep in mind 9 performed really well
            good_draw.append([m])
            good.append(m) 
    return good,good_draw

def reproject_npz(Pano_dir='./track_panorama.png', samples_dir="./car_images",
                  orig_locations_dir="vehicle_locations.json", x_1=0, y_1=0,
                  x_origin=0, y_origin=0, ratio=0):
    start = time.time()
    frame_data = []  # store detections for this frame
    results = {}  # dictionary to store results as before

    RANSAC_REPROJ_THRESH = 2.0   # px; homography ransac reprojection threshold

    img2 = cv2.imread(Pano_dir, cv2.IMREAD_GRAYSCALE)  # panorama image

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find panorama keypoints and descriptors with SIFT 
    kp2, des2 = sift.detectAndCompute(img2, None)  # this will be used recurringly

    f_frames = glob.glob(os.path.join(samples_dir, "*.jpg"))  # read in all frame paths

    with open(orig_locations_dir, "r") as f:
        data = json.load(f)

    for f_frame in f_frames:
        img1 = cv2.imread(f_frame)  # read in current frame
        frame_name = os.path.basename(f_frame)

        x1, y1, x2, y2 = data[frame_name][0]["box"]
        x_center, y_center = data[frame_name][0]["center"]

        kp1, des1 = sift.detectAndCompute(img1, None)  # compute keypoints

        # BFMatcher with KNN
        good, good_draw = Knn_matches(des1, des2, 2)

        if len(good) < 10:
            raise RuntimeError("Not enough good matches to compute homography")

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESH)
        if H is None:
            raise RuntimeError("Homography estimation failed.")

        corners = np.array([[[x1, y2]], [[x2, y2]], [[x2, y1]], [[x1, y1]]], dtype=np.float32)
        center = np.array([[[x_center, y_center]]], dtype=np.float32)

        # Transform corners and center to panorama coordinates
        projected_corners = cv2.perspectiveTransform(corners, H)
        projected_center = cv2.perspectiveTransform(center, H)

        # Adjust for origin
        projected_center[0, 0, 0] -= x_1
        projected_center[0, 0, 1] -= y_1

        real_world_x = (projected_center[0, 0, 0] - x_origin) * ratio
        real_world_y = -(projected_center[0, 0, 1] - y_origin) * ratio
        rw_center = np.array([[[real_world_x, real_world_y]]], dtype=np.float32)

        # Store results in a dictionary
        results[frame_name] = {
            "reprojected_corners": projected_corners,
            "reprojected_center_occup": projected_center,
            "real_world_coords": rw_center
        }

    # --- Save results to npz file ---
    npz_dict = {}
    for k, v in results.items():
        npz_dict[f"{k}_corners"] = v["reprojected_corners"]
        npz_dict[f"{k}_center_occ"] = v["reprojected_center_occup"]
        npz_dict[f"{k}_real"] = v["real_world_coords"]

    np.savez("reprojected_corners.npz", **npz_dict)

    print(f"Total time to compute: {time.time() - start:.2f} seconds")
