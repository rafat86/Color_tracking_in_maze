# Assignment 02 #
# Color Detection and Tracking #
# Ra'fat Naserdeen #

import cv2
import numpy as np

width = 640  # Image and Video Width
height = 480  # Image and Video Height

maze = cv2.imread('maze.jpg', -1)
image1 = cv2.resize(maze, (width, height))

start_drawing = False


def detect_green_ball(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)  # Threshold the HSV image to get only green colors

    # Find contours in the green mask
    contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)  # Get the largest contour

        if cv2.contourArea(largest_contour) >= 1000:

            m = cv2.moments(largest_contour)
            if m["m00"] != 0:
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])
                return cx, cy

    return None, None


def draw(size):
    # Draw a green square around the ball on the frame
    square_size = size
    x1, y1 = cx - square_size // 2, cy - square_size // 2
    x2, y2 = cx + square_size // 2, cy + square_size // 2

    # Draw a green square around the ball on image1
    x1_image1, y1_image1 = int(cx / width * image1.shape[1]) - square_size // 2, \
        int(cy / height * image1.shape[0]) - square_size // 2
    x2_image1, y2_image1 = x1_image1 + square_size, y1_image1 + square_size

    return cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2), cv2.rectangle(image1,
            (x1_image1, y1_image1), (x2_image1, y2_image1), (0, 255, 0), 2)


cap = cv2.VideoCapture(0)  # Open webcam

# Set resolution of the webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip Video horizontally

    cx, cy = detect_green_ball(frame)   # Detect green ball and get its centroid

    if cx is not None and cy is not None:

        if cx in range(80, 100) and cy in range(115, 130):
            start_drawing = True

        if start_drawing:
            draw(4)
            if cx in range(540, 575) and cy in range(350, 355):
                break

        else:
            image1 = cv2.resize(maze, (width, height))  # Clear the contents of image1
            draw(4)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)   # Check for yellow ball
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

    if cv2.countNonZero(mask_yellow) > 1000:
        print("Yellow ball detected. Stopping video.")
        break

    cv2.imshow('Frame', frame)
    cv2.imshow('Movement Line', image1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
