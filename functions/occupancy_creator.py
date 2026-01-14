import cv2
import numpy as np

# Load the image

def occupancycreator(imgdir = "track_panorama.png"):
    image = cv2.imread(imgdir)
    # Scale factor > 1.0 makes image brighter (like increasing exposure)
    exposure_factor = 1.3
    bright = np.clip(image * exposure_factor, 0, 255).astype(np.uint8)

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a window
    cv2.namedWindow("Trackbar Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Trackbar Controls", 600, 400)  # width, height


    # Nothing function for trackbar
    def nothing(x):
        pass

    # Create trackbars for lower and upper HSV
    cv2.createTrackbar("Lower H", "Trackbar Controls", 0, 180, nothing)
    cv2.createTrackbar("Lower S", "Trackbar Controls", 0, 255, nothing)
    cv2.createTrackbar("Lower V", "Trackbar Controls", 0, 255, nothing)
    cv2.createTrackbar("Upper H", "Trackbar Controls", 180, 180, nothing)
    cv2.createTrackbar("Upper S", "Trackbar Controls", 68, 255, nothing)
    cv2.createTrackbar("Upper V", "Trackbar Controls", 92, 255, nothing)

    while True:
        # Get current positions of trackbars
        l_h = cv2.getTrackbarPos("Lower H", "Trackbar Controls")
        l_s = cv2.getTrackbarPos("Lower S", "Trackbar Controls")
        l_v = cv2.getTrackbarPos("Lower V", "Trackbar Controls")
        u_h = cv2.getTrackbarPos("Upper H", "Trackbar Controls")
        u_s = cv2.getTrackbarPos("Upper S", "Trackbar Controls")
        u_v = cv2.getTrackbarPos("Upper V", "Trackbar Controls")

        # Set lower and upper HSV range
        lower_wall = np.array([l_h, l_s, l_v])
        upper_wall = np.array([u_h, u_s, u_v])

        # Threshold to get only walls
        mask_wall = cv2.inRange(hsv, lower_wall, upper_wall)
        mask_tiles = cv2.bitwise_not(mask_wall)

        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        #cv2.namedWindow("Brightness Adjusted", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Wall Mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Tile Mask", cv2.WINDOW_NORMAL)

        # Show results
        cv2.imshow("Original", image)
        #cv2.imshow("Brightness Adjusted", bright)
        cv2.imshow("Wall Mask", mask_wall)
        cv2.imshow("Tile Mask", mask_tiles)

        # Break on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite("tile_mask.png", mask_tiles)
            cv2.imwrite("mask_wall.png",mask_wall )
            print("Tile mask saved as tile_mask.png")
            break

    cv2.destroyAllWindows()