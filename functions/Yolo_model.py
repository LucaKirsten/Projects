from ultralytics import YOLO

# Load your trained model
model = YOLO("./best.pt")
"""
# Run prediction on an image
results = model.predict(source="./Samples_car_UW/frame_000000.jpg", 
                        conf=0.6, 
                        imgsz=640,
                        device="cpu",
                        save=True)

runs in 0.14 seconds rougly 
"""


# Or on a folder of images
results = model.predict(source="./Samples_car_UW", 
                        conf=0.6, 
                        imgsz=640,
                        device="cpu",
                        save=True,
                        save_txt=True
                        )

"""
# for video
results = model.predict(source = "./video_retakes/track_1_car_UW_14_09.mp4", 
                        conf=0.6, 
                        imgsz=640,
                        device="cpu",
                        save= True, 
                        save_txt = True)
"""
