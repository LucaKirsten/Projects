"""
file used to determine origin of occupancy grid from yaml 
pixel value is kept 
image is cropped to left most, right most, highest and lowest pixels 
resize to 1080 1920
the origin is then recalculated from the bottom left 0,0 pixel 

same is then done for the self created occupancy grid
resized to 1080 1920

origin is then transposed onto self created occupancy grid
pixel coordinates of car centre are then calculated in distance from occupancy grid

"""
import yaml
import cv2
import matplotlib.pyplot as plt

# Load YAML
def origin_finder(yaml_path = "sep2.yaml", img_path = "sep2.png", occup = "tile_mask_edited_cropped.png", ratio = 0):
    with open(yaml_path, "r") as f:
        map_yaml = yaml.safe_load(f)

    resolution = map_yaml["resolution"] # size of each pixel in real world distances
    origin = map_yaml["origin"]  # [x_origin, y_origin, theta]

# Load image to get size
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    print(f"Image shape: {img.shape}, dtype: {img.dtype}")

    height, width = img.shape[:2]


# Origin in world coordinates
    x_world, y_world = origin[0], origin[1]

# Convert to pixel coordinates
# Assuming origin corresponds to bottom-left of image
# 1 px: 0.05 m thus real world/res = pixel coord
    pixel_x = abs(x_world) / resolution
    pixel_y = abs(y_world) / resolution

# If origin is bottom-left, y needs flipping for image coordinates
# y max pixel is at bottom
# thus max y pixel - pixel distance = pixel position 
    pixel_y_image = height - pixel_y

    print(f"Origin pixel coordinates in image: ({pixel_x}, {pixel_y_image})")

# occupancy threshold represents probability that pixel is unoccupied thus if above 0.65*255 ->
# unoccupied and should be completely white

    _, binary = cv2.threshold(img, int(map_yaml['occupied_thresh'] * 255), 255, cv2.THRESH_BINARY) 

#find all white pixels 
    coords = cv2.findNonZero(binary)  # all white pixel coordinates
    x, y, w, h = cv2.boundingRect(coords)

# crop image to size of y bottom plus hight and x left plus width
    cropped = img[y:y+h, x:x+w]

    cr_h, cr_w = cropped.shape[:2]
    cropped_x_origin = pixel_x - x
    cropped_y_origin = pixel_y_image - y # pixel from top thus distance from bottom is 
    cx, cy = int(round(cropped_x_origin)), int(round(cropped_y_origin))

    if len(cropped.shape) == 2:
        cropped_color = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
    else:
        cropped_color = cropped.copy()

        # Draw a visible circle (e.g., radius 10, red color)
    cv2.circle(cropped_color, (cx, cy), radius=10, color=(0, 0, 255), thickness=2)

    # Optionally, mark the exact origin with a small filled center
    cv2.circle(cropped_color, (cx, cy), radius=2, color=(0, 255, 0), thickness=-1)

    # Show and save
    plt.imshow(cropped_color)
    plt.title("Origin Marked")
    plt.show()
    cv2.waitKey(0)

    print(f"Originpixel coordintates in cropped image: ({cropped_x_origin}, {cropped_y_origin})")

#thus real world distance from bottom left is now 
    print(f"dim of cropped image: width :{cr_w}, height:{cr_h}")

    new_world_x = -cropped_x_origin*resolution # real world x 
    new_world_y = -(cr_h-cropped_y_origin)*resolution # real world y

    print(f"real world coordinates in new cropped image with bottom left as zero from pixels: {-cropped_x_origin*resolution}, {-(cr_h-cropped_y_origin)*resolution}")

# now for my occupancy grid cropped

    self_occupancy = cv2.imread(occup, cv2.IMREAD_GRAYSCALE)
#cv2.imshow("occupancy grid self",self_occupancy)
#cv2.waitKey(500)
    coords_1 = cv2.findNonZero(self_occupancy)  # all white pixel coordinates
    x_1, y_1, w_1, h_1 = cv2.boundingRect(coords_1)

# since no resizing the img distancing stays constant from ratio.py

    cropped_self = self_occupancy[y_1:y_1+h_1, x_1:x_1+w_1]
    print(f"shape of cropped panorama: height -> {cropped_self.shape[0]} width -> {cropped_self.shape[1]}")
    print(f"ratio : {ratio}")
    dimh, dimw = cropped_self.shape[:2]
    px_origin = round((abs(new_world_x)/ratio))
    py_origin = round(dimh - (abs(new_world_y)/ratio))
    

    
    print(f" px origin: {px_origin}, py origin{py_origin}: ")

    if len(cropped_self.shape) == 2:
        cropped_self_color = cv2.cvtColor(cropped_self, cv2.COLOR_GRAY2BGR)
    else:
        cropped_self_color = cropped_self.copy()
    # Draw a visible circle (e.g., radius 10, red color)
    cv2.circle(cropped_self_color, (px_origin, py_origin), radius=10, color=(0, 0, 255), thickness=2)

    # Optionally, mark the exact origin with a small filled center
    cv2.circle(cropped_self_color, (px_origin, py_origin), radius=2, color=(0, 255, 0), thickness=-1)

    # Show and save
    plt.imshow(cropped_self_color)
    plt.title("Origin Marked")
    plt.show()
    cv2.waitKey(0)


    """
    return new given occupancy grid origin 
    return new own occupancy grid origin

    dont need to return cutoff x and y for given occupancy as particle filter gives distance in terms of real world 
    need to return cutoff x,y pixels for self as need to transform pixels to distance from origin on cropped occupancy grid

    """
    print(y_1)
    # to compute origin in non cropped occupancy grid we have:
    
    # thus to determine the distance from the old image bottom left we subtract the difference further from the current relative position
    print(f"original origin on non cropped self occup: x {x_1 + px_origin}, y : {y_1 + py_origin}")
    #now calculate the distance to the pixel in real world
    
    return new_world_x, new_world_y, cropped_x_origin, cropped_y_origin, px_origin, py_origin, x_1, y_1