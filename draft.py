import cv2
import numpy as np

# Set up the capture object
cap = cv2.VideoCapture('gaKhanh2.mp4')

# Check if the camera opened successfully
if not cap.isOpened():
    raise Exception("Could not open video device")
while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if not ret:
        raise Exception("Could not read frame from video device")

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply edge detection to the image
    edges = cv2.Canny(gray, 190, 255)

    # Find contours in the edges image
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Create arrays to store the x and y coordinates of the left and right lines
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    # Loop through each contour that was detected
    for contour in contours:
        # Fit a line to the contour
        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        # Calculate the slope of the line
        slope = vy / vx
        # If the slope is positive, it is a left line
        if slope > 0:
            left_line_x.extend(x)
            left_line_y.extend(y)
        # If the slope is negative, it is a right line
        else:
            right_line_x.extend(x)
            right_line_y.extend(y)

    # Fit a polynomial to the left and right lines
    left_line_coefficients = np.polyfit(left_line_y, left_line_x, 1)
    right_line_coefficients = np.polyfit(right_line_y, right_line_x, 1)

    # Calculate the x-coordinates of the left and right lines at the top and bottom of the image
    y1 = 0
    y2 = frame.shape[0]
    left_x1 = int(np.polyval(left_line_coefficients, y1))
    left_x2 = int(np.polyval(left_line_coefficients, y2))
    right_x1 = int(np.polyval(right_line_coefficients, y1))
    right_x2 = int(np.polyval(right_line_coefficients, y2))

    # Draw the left and right lines on the image
    cv2.line(frame,(left_x1,y1),(left_x2,y2),(255,0,0),5)
    cv2.line(frame,(right_x1,y1),(right_x2,y2),(255,0,0),5)

    # Calculate mid line 
    mid_x1=int((left_x1+right_x1)/2)
    mid_x2=int((left_x2+right_x2)/2)
    cv2.line(frame,(mid_x1,y1),(mid_x2,y2),(0,255,0),5)

    # Display the image with the lines drawn on it
    cv2.imshow('frame', frame)

    # Wait for a key press and then release the capture object and destroy all windows
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()