import numpy as np
import cv2
import time
import math
from function import *
import pygame
import sys
cap = cv2.VideoCapture('gaKhanh.mp4')
counting = 0
fps = 0
in1 = 37 #left 
in2 = 35 #left //forward
in3 = 33 #right //forward
in4 = 31 #right 
leftSpeed = 120
rightSpeed = 120
midline_x1 = None
midline_x2 = None
lane_left,lane_right=None,None
pygame.init()
timecheckpoint = time.time()
screen = pygame.display.set_mode([640, 480])
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            sys.exit()
    counting+=1
    if counting == 10:
        fps = 10 /  (time.time() - timecheckpoint)
        timecheckpoint = time.time()
        counting = 0
        print(fps)
    # Capture frame from camera
    ret, img = cap.read()
    if not ret:
        print('video ended')
        break
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Use global threshold based on grayscale intensity.
    threshold = cv2.inRange(gray, 190, 255)
    
    width = threshold.shape[1]
    height = threshold.shape[0]
    roi_vertices = np.array([[[width *0/10, height],
                          [width *10/10, height],
                          [width * 10 / 10, height * 3 / 10],
                          [width * 0 / 10, height * 3 / 10]]],dtype=np.int32)

    # Defining a blank mask.
    mask = np.zeros_like(threshold)   

    # Defining a 3 channel or 1 channel color to fill the mask.
    if len(threshold.shape) > 2:
        channel_count = threshold.shape[2]  # 3 or 4 depending on the image.
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # Filling pixels inside the polygon.
    cv2.fillPoly(mask, roi_vertices, ignore_mask_color)

    # Constructing the region of interest based on where mask pixels are nonzero.
    roi = cv2.bitwise_and(threshold, mask)
    # Smooth with a Gaussian blur.
    kernel_size = 3
    roi_blur = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)

    # Perform Edge Detection.
    low_threshold = 100
    high_threshold = 255
    canny_blur = cv2.Canny(roi_blur, low_threshold, high_threshold)

    lines = cv2.HoughLinesP(canny_blur, 1, np.pi/180, 120, np.array([]), minLineLength=150, maxLineGap=20)

    # Draw all lines found onto a new image.
    hough = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
    draw_lines(hough, lines)

    # Define bounds of the region of interest.
    roi_upper_border = int(img.shape[0] * 3 / 10)
    roi_lower_border = img.shape[0]

    # Create a blank array to contain the (colorized) results.
    lanes_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)

    # Use above defined function to identify lists of left-sided and right-sided lines.
    
    lines_left, lines_right = separate_left_right_lines(lines,img.shape[1]/2)

    # Use above defined function to extrapolate the lists of lines into recognized lanes.
    lane_left = extrapolate_lines(lines_left, roi_upper_border, roi_lower_border)
    lane_right = extrapolate_lines(lines_right, roi_upper_border, roi_lower_border)

    if lane_left is not None: #and not np.isnan(lane_left).any():
        draw_lines(lanes_img, [[lane_left]], thickness = 10)
        x1_left,y1_left,x2_left,y2_left = lane_left
        if y1_left > y2_left:
            x1_left,y1_left,x2_left,y2_left = x2_left,y2_left,x1_left,y1_left
                    

    if lane_right is not None:# and not np.isnan(lane_right).any():
        draw_lines(lanes_img, [[lane_right]], thickness = 10)
        x1_right,y1_right,x2_right,y2_right = lane_right
        if y1_right > y2_right:
            x1_right,y1_right,x2_right,y2_right = x2_right,y2_right,x1_right,y1_right

    if lane_left is not None and lane_right is not None:# and not np.isnan(lane_left).any() and not np.isnan(lane_right).any():
        midline_x1=int((x1_left + x1_right)/2)
        midline_x2=int((x2_left + x2_right)/2)

        cv2.line(lanes_img,(midline_x1,y1_left),(midline_x2,y2_left),(0,255,0),10)
    
    center_point = (int(width/2), int(height/2))
    
    if midline_x1 is not None:
        dis, proj_point = distance(midline_x1, y1_left, midline_x2, y2_left, center_point[0], center_point[1])
        cv2.line(lanes_img,center_point, proj_point,(0,255,255),10)
        cv2.line(lanes_img,center_point, (midline_x2,y2_left),(0,255,255),10)
        cv2.line(lanes_img,center_point, (center_point[0],height),(0,255,255),10)

        if center_point[0] > proj_point[0]: 
            cv2.putText(lanes_img, "Left: " + str(round(dis,2)), center_point, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color= (255, 255, 255), lineType=2)
        else:
            cv2.putText(lanes_img, "Right: "+ str(round(dis,2)), center_point, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color= (255, 255, 255), lineType=2)

    image_annotated = cv2.addWeighted(img, 0.8, lanes_img, 1, 0)
    
    cv2.putText(image_annotated, 'Fps: '+str(round(fps,2)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color= (255, 255, 255), lineType=2)
    result = cv2.cvtColor(image_annotated, cv2.COLOR_BGR2RGB)
    cv2.imshow('canny',canny_blur)
    cv2.imshow('res',image_annotated)

    # Convert the result to a Pygame surface
    result = np.rot90(result)
    result = result[::-1,:,:]
    result = pygame.surfarray.make_surface(result)

    # Display the result on the Pygame screen
    screen.blit(result, (0, 0))
    pygame.display.update()
    # Display the results, and save image to file.
    image_annotated = cv2.cvtColor(image_annotated, cv2.COLOR_BGR2RGB)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    