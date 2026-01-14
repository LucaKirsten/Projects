import os
import json

def location_store(labels_folder = "./runs/detect/predict/labels",images_folder = "./runs/detect/predict" ):
    image_width = 1064
    image_height = 1920

    results = {}  # dictionary to store results

    # Loop over all label files
    for label_file in os.listdir(labels_folder):
        if not label_file.endswith(".txt"):
            continue

        label_path = os.path.join(labels_folder, label_file)
        with open(label_path, "r") as f:
            lines = f.readlines()

        image_name = label_file.replace(".txt", ".jpg")  # corresponding image
        

        frame_data = []  # store detections for this frame

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts)

            # Convert YOLO normalized coords to pixel coordinates
            x_center = x_center_norm * image_width
            y_center = y_center_norm * image_height
            box_width = width_norm * image_width
            box_height = height_norm * image_height

            # Convert center coordinates to corner coordinates
            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            # Store detection as dictionary
            frame_data.append({
                "class": int(class_id),
                "center": [x_center, y_center],
                "box": [x1, y1, x2, y2]
            })

        # Save detections for this frame under the image name
        results[image_name] = frame_data

    # Write everything to one JSON file
    output_file = "vehicle_locations.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved results to {output_file}")