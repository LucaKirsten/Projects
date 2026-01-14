import cv2
import numpy as np
import glob
import os


DIM=(1920, 1080)
K= np.array([
    [940.11, 0, 959.07],
    [0, 937.14, 541.26],
    [0, 0, 1]
])
D=np.array([0.3908, -0.4067, 0.5585, 0.0083])

"""

# --- Ultra Wide Lens ---
K_ultra = np.array([
    [940.11, 0, 959.07],
    [0, 937.14, 541.26],
    [0, 0, 1]
])

D_ultra = np.array([0.3908, -0.4067, 0.5585, 0.0083])

# --- Wide Lens ---
K_wide = np.array([
    [967.09, 0, 968.35],
    [0, 965.12, 537.53],
    [0, 0, 1]
])

D_wide = np.array([0.0394, 0.0251, 0.0323, -0.0434])

"""
#note track was sampled at 8 frames and then undistorted 
#undistort(img, balance=0.5, shift_x=-200, idx=index)

# note dim3 = dim2 in function call 
def undistort(temp, balance=1, dim2=None, dim3=None, shift_x=0, idx=0, OutputPath="./undistorted_images", rotate = True):
    if rotate:
        img= cv2.rotate(temp, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        img = temp
    dim1 = img.shape[:2][::-1]

    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio"

    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1

    scaled_K = K * dim1[0] / DIM[0]
    scaled_K[2][2] = 1.0

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        scaled_K, D, dim2, np.eye(3), balance=balance)

    # Shift principal point horizontally if needed
    new_K[0, 2] += shift_x

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        scaled_K, D, np.eye(3), new_K, dim2, cv2.CV_16SC2)

    undistorted_img = cv2.remap(
        img, map1, map2, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    filename = os.path.join(OutputPath, f"frame_{idx:06d}.jpg")
    rotated = cv2.rotate(undistorted_img, cv2.ROTATE_90_CLOCKWISE)
    
    cropped_img = rotated[:, 8:1072]  # keep all rows, slice columns from 75 to 1845
    
    cv2.imwrite(filename, cropped_img)

if __name__ == '__main__':
    
    ext= "*.jpg"
    fnames =  glob.glob(os.path.join("../stitching_images", "*.jpg")) # image extention and folder containing the sampled images from video

    if fnames is None: 
        raise Exception("dir is wrong or no images of type{ext[1:]}")
    index = 0
    for f in fnames:
        img = cv2.imread(f)
        undistort(img, balance=0.5, shift_x=0, idx=index, OutputPath="./stitching_img_undistorted")
        index+=1