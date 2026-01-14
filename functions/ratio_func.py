"""
input:
panorama
image with checkerboard (1)
image without checkerboard (2)

find pixel cooridinates of corners of one square on checkerboard (3) 
    - let y1 be max(y1,y2) 
    - let x1 be max(x1,x2)

find homography between (2) and panorama 
transpose (3) onto panorama
(x1-x2 + x3-x4)/2 = xdim in real world
(y1-y2 + y3-y4)2 = ydim in real world 

"""

import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
from functions import undistort as und

def reprojector(tr,br,tl,bl, img, H):
    
    corners = np.array([[tl],           # top-left
                        [tr],          # top-right
                        [br],         # bottom-right
                        [bl]],         # bottom-left
                    dtype=np.float32)

    # Transform corners to panorama coordinates
    projected_corners = cv2.perspectiveTransform(corners, H)
    # print(projected_corners)

    # Convert to int for drawing
    projected_corners_int = np.int32(projected_corners)
    # print(projected_corners_int)

    # Draw polygon on panorama
    img_with_box = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # convert panorama to color
    cv2.polylines(img_with_box, [projected_corners_int], isClosed=True, color=(0, 255, 0), thickness=3)

    # Show the result
    #plt.imshow(cv2.cvtColor(img_with_box, cv2.COLOR_BGR2RGB))
    #plt.title("Smaller image projected onto panorama")
    #plt.show()

    return projected_corners

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
        if m.distance < 0.7 * n.distance:
            good_draw.append([m])
            good.append(m) 
    return good,good_draw

def ratio_calc(wide = False):
    # corner detection criteria
    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001) # type -> # iterations, epsilon for accuracy

    CHECKERBOARD = [4,6]
    imgpoints = [] # 2D image coordinates 

    if wide:
        temp1= cv2.imread("./ratio_1/frame_000000.jpg")
        temp2= cv2.imread("./ratio_2/frame_000000.jpg")

        und.undistort(temp=temp1, OutputPath="./ratio_1", rotate=False)
        und.undistort(temp=temp2, OutputPath="./ratio_2", rotate=False)

        img1 = cv2.imread("./ratio_1/frame_000000.jpg") # with checkerboard
        img2 = cv2.imread("./ratio_2/frame_000000.jpg") # without checkerboard
    else:
        img1 = cv2.imread("./ratio_1/frame_000000.jpg") # with checkerboard
        img2 = cv2.imread("./ratio_2/frame_000000.jpg") # without checkerboard
        
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # convert image to gray scale image for corner finding

        # find cornerns 
        # flags -> 1) converts to balck and white rahter than fixed theshold
        #       -> 2) normalises image gamma before applying fixed or adaptive threshholding
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True: # if corners were found

        #find more exact corner points after they are detected 
        corners2 =cv2.cornerSubPix(gray, corners, (3,3), (-1, -1), criteria) 
        imgpoints.append(corners2)

        #cv2.drawChessboardCorners(img1, CHECKERBOARD, corners2, ret)
        #cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        #cv2.imshow('img', img1)
        #cv2.waitKey(0)

    cv2.destroyAllWindows()

    start = time.time()

    RANSAC_REPROJ_THRESH = 2.0   # px; homography ransac reprojection threshold

    img1 = img2 
    img2 = cv2.imread('./track_panorama.png',cv2.IMREAD_GRAYSCALE) # panorama image

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)

    start_1 = time.time()
    kp2, des2 = sift.detectAndCompute(img2,None)
    print(f"time to compute points for panorama : {time.time() - start_1}")


    start_matching = time.time()

    # BFMatcher with default params
    # brute force matcher struggles with large data sets, thus will implement both BF matcher and FLANN 

    # bf implementation with KNN
    # good, good_draw = Knn_matches(des1, des2, 2)

    # normal bf implementation 
    # good, good_draw = normal_BF(des1, des2)

    # FLANN implementation

    good, good_draw = Flann_matcher(des1, des2, 2)
    print(f"time to match : {time.time() - start_matching}")


    # cv.drawMatchesKnn expects list of lists as matches thus good_draw and good.
    #img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good_draw,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # cv
    print(f"time taken to process images: {time.time() - start}")
    #plt.imshow(img3),plt.show()

    # now mathes are computed, need to warp smaller image to panorama plane and then project point where car is
    # replicate non mathcing feature by drawing a large circle, finding bouding box and inputting centre on panorama 

    # need enough matches to compute homography 

    if len(good) < 10:
        raise RuntimeError("Not enough good matches to compute homography")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)  # frame points in shape (N, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)  # panorama points in shape (N, 1, 2) 

    # -1 counts number of points 
    #  1 is a convention additional dimension
    #  2 is how many points to group together ( take 2 elements from flatarray per point )

    # print(f"{src_pts}")
    # note differnce between this and calibration which uses (N,1,3) because this is not a 3d transformation and images are in same plane 

    H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESH)
    if H is None:
        raise RuntimeError("Homography estimation failed.")

    print(corners2)

    tr = corners2[0,0]
    br = corners2[1,0]
    tl = corners2[4,0]
    bl = corners2[5,0]
    print(f"{tr}, {br}, {tl}, {bl}")

    projected = reprojector(tr,br,tl,bl, img2, H) #calculate and visualize corners on panorama

    brp = projected[0,0]

    blp = projected[1,0]

    tlp = projected[2,0]

    trp = projected[3,0]

    x_width = round(np.abs((trp[0] - tlp[0]) + (brp[0] - blp[0]))/2)
    y_height = round(np.abs((trp[1] - brp[1]) + (tlp[1] - blp[1]))/2)

    print(f"checkerboard square is ({x_width},{y_height} )") 
    rows = 6
    cols = 4
    x_widths = []
    y_heights = []
    y_norms = []
    x_norms = []

    for i in range(rows - 1):       # 0 → 4
        for j in range(cols - 1):   # 0 → 2
            idx_tl = i * cols + j
            idx_tr = i * cols + (j + 1)
            idx_bl = (i + 1) * cols + j
            idx_br = (i + 1) * cols + (j + 1)

            tl = corners2[idx_tl, 0]
            tr = corners2[idx_tr, 0]
            bl = corners2[idx_bl, 0]
            br = corners2[idx_br, 0]

            projected = reprojector(tr, br, tl, bl, img2, H)

            brp = projected[0, 0]
            blp = projected[1, 0]
            tlp = projected[2, 0]
            trp = projected[3, 0]

            x_width = round(np.abs((trp[0] - tlp[0]) + (brp[0] - blp[0])) / 2)
            hor_dist = np.linalg.norm((trp[0] - tlp[0])+(brp[0] - blp[0]))/2
            vert_dist = np.linalg.norm((trp[1] - brp[1])+(tlp[1] - blp[1]))/2
            y_height = round(np.abs((trp[1] - brp[1]) + (tlp[1] - blp[1])) / 2)

            if y_height == 0:   # guard to prevent division by zero
                continue

            x_widths.append(x_width)
            y_heights.append(y_height)
            x_norms.append(hor_dist)
            y_norms.append(vert_dist)

    x_mean = np.mean(x_widths)
    y_mean = np.mean(y_heights)
    x_norm_mean = np.mean(x_norms)
    y_norm_mean = np.mean(y_norms)
    y_mean_ratio = 0.1/y_mean
    x_mean_ratio = 0.1/x_mean


    ratio_x = 0.1/x_width
    ratio_y = 0.1/y_height
    #print(f"x space = {x_width}")
    #print(f"y height = {y_height}")
    print(f"ratio x = {ratio_x}")
    print(f"ratio y = {ratio_y}")

    #print(f"average euclidean distance between corners: x: {x_norm_mean}, y: {y_norm_mean}")
    print(f"average euclidean distance ratio between corners: x: {0.1/x_norm_mean}, y: {0.1/y_norm_mean}")
    #print(f"mean average of all squares x: {x_mean}, y: {y_mean}")
    print(f"mean average of all squares x: {x_mean_ratio}, y: {y_mean_ratio}")
    return ratio_x
