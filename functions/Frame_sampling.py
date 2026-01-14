import cv2
import numpy as np
import os
import time

def Frame_Sampling(VidPath = "./video_retakes/track_1_W_11_09.mp4", OutputPath = "./sampled_images", SampleRate = 8, rotate = True, resize=False):
    """
    Vidpath: sample video 
    OutputPath: where frames are to be stored
    SampleRate: nth frame to be sampled
    resize: if image is (3840,2160) as this is to many pixels and increases computational time
    rotate: output images need to be (1080,1920)

    """
    start = time.time()
    
    imgExt = "png"

    cap = cv2.VideoCapture(VidPath)
    
    if not cap.isOpened(): 
        raise RuntimeError(f"no video found at file location: {VidPath}")
    else: 
        print("Video opened successfully")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"fps={fps:.3f}, approx frames={frame_count}")

    idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % SampleRate == 0 :
            filename = os.path.join(OutputPath, f"frame_{idx:06d}.{imgExt}")

            if resize:
                resized = cv2.resize(frame, (1080, 1920), interpolation=cv2.INTER_AREA)
            else:
                resized = frame

            if rotate:
                rotated = cv2.rotate(resized, cv2.ROTATE_90_CLOCKWISE)
            else:
                rotated = resized

            # note it halfs the time when using jpg            
            cv2.imwrite(filename.replace(".png", ".jpg"), rotated, [int(cv2.IMWRITE_JPEG_QUALITY), 95]) # jpg is faster as it compresses image more might result in lost pixels but not to an extent that it would cause error with object matching
            saved+=1
        idx+=1
    cap.release()
    print(f"Saved {saved} frames to {OutputPath}")
    print(f" time elapsed: {time.time() - start}")

if __name__ == '__main__':
    Frame_Sampling( rotate=True, OutputPath="./track_1_UW",VidPath="./video_retakes/track_1_UW_14_09.mp4", SampleRate=14)