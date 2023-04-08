import numpy as np
import cv2
import time
import math
import Jetson.GPIO as GPIO
import sys
from function import *

cap = cv2.VideoCapture(0)
counting = 0
fps = 0
in1 = 37 #left 
in2 = 35 #left //forward
in3 = 33 #right //forward
in4 = 31 #right 
leftSpeed = 120
rightSpeed = 120
midline_x1 = None
lane_left,lane_right=None,None
src_ref=np.float32([(250,0),(1150,0),(0,720),(1280,720)])
dst_ref=np.float32([(0,0), (1280, 0), (0,720), (1280,720)])
GPIO.setmode(GPIO.BOARD)
GPIO.setup(in1,GPIO.OUT)
GPIO.setup(in2,GPIO.OUT)
GPIO.setup(in3,GPIO.OUT)
GPIO.setup(in4,GPIO.OUT)
GPIO.setwarnings(False)
timecheckpoint = time.time()
cal_dir='./cal_pickle.p'

with open(cal_dir, mode='rb') as f:
    file = pickle.load(f)
    mtx = file['mtx']
    dist = file['dist']
while True:
    counting+=1
    if counting == 10:
        fps = 10 /  (time.time() - timecheckpoint)
        timecheckpoint = time.time()
        counting = 0
        print(fps)
    # Capture frame from camera
    ret, img = cap.read()
    distort = cv2.undistort(img, mtx, dist, None, mtx)
    if not ret:
        print('video ended')
        break
    # Use global threshold based on grayscale intensity.
    lanes_img,leftLines,rightLines,midline_x1 = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
    nextMove(img.shape,leftLines,rightLines,midline_x1)
    
    width = img.shape[1]
    height = img.shape[0]
    roi_vertices_down = np.array([[[width *0/10, height],
                          [width *10/10, height],
                          [width * 10 / 10, height * 5 / 10],
                          [width * 0 / 10, height * 5 / 10]]],dtype=np.int32)
    
    center_point = (int(img.shape[1]/2), int(img.shape[0]/2))
    M = cv2.getPerspectiveTransform(src_ref, dst_ref)
    center_point_warp = point_warp(center_point,M)

    warp = perspective_warp(distort, src_ref, dst_ref)
    image_annotated,lanes_img = make_way('1',img=img,roi_vertices=roi_vertices_down,threshold=(200,255),minLineLength=100,lanes_img=lanes_img,center_point=center_point_warp)  
    

    cv2.putText(image_annotated, 'Fps: '+str(round(fps,2)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color= (255, 255, 255), lineType=2)


    #show on pygame window
    result = cv2.cvtColor(image_annotated, cv2.COLOR_BGR2RGB)
    result = np.rot90(result)
    result = result[::-1,:,:]

    # Display the results, and save image to file.
    image_annotated = cv2.cvtColor(image_annotated, cv2.COLOR_BGR2RGB)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    