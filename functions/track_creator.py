import cv2 
import glob
import time
import os 

def Image_stitcher(ImgDir = "./sampled_images", ImgExt = "*.jpg",): 
    start = time.time()
    print("Starting image stitching")
    images_f = glob.glob(os.path.join(ImgDir, ImgExt))
    images = []

    if not images_f:
        raise RuntimeError(f"no jpg images found in directory: {ImgDir}")
    
    for fname in images_f:
        img = cv2.imread(fname)
        images.append(img)


    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)

    status, pano = stitcher.stitch(images)
    
    if status != cv2.Stitcher_OK:
        raise RuntimeError(f"stitching failed with status: {status}") 
    
    cv2.imwrite("track_panorama.png", pano)
    print(f"successfully stitched image in : {time.time() - start}")
    
#Image_stitcher(ImgDir="../stitching_img_undistorted")