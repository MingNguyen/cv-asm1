import cv2
import time
import numpy as np
import math
import pickle
def perspective_warp(img, src, dst):
    img_size = (1280,720)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    return warped

def inv_perspective_warp(img, src, dst):
    img_size = (1280,720)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    return warped
def point_warp(point, M):
    # convert the point to homogeneous coordinates
    homo = np.array([point[0], point[1], 1]).reshape(-1, 1)
    # apply perspective transform to the point
    warped_point_homo = M @ homo
    warped_point = (warped_point_homo[:2] / warped_point_homo[2]).astype(int)
    return warped_point

def draw_lines(img, lines, color = [255, 0, 0], thickness = 2):
    """Utility for drawing lines."""
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
def separate_left_right_lines(lines,imgMid):
    """Separate left and right lines depending on the slope."""
    left_lines = []
    right_lines = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if y1 > y2:
                    x1,y1,x2,y2 = x2,y2,x1,y1
                    
                if x2 < imgMid: # Negative slope = left lane.
                    left_lines.append([x1, y1, x2, y2])
                elif x2 > imgMid: # Positive slope = right lane.
                    right_lines.append([x1, y1, x2, y2])
    return left_lines, right_lines
def cal_avg(values):
    """Calculate average value."""
    if not (type(values) == 'NoneType'):
        if len(values) > 0:
            n = len(values)
        else:
            n = 1
        return sum(values) / n
def extrapolate_lines(lines, upper_border, lower_border):
    
    """Extrapolate lines keeping in mind the lower and upper border intersections."""
    slopes = []
    consts = []
    
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            if x1 == x2: return None
            slope = (y1-y2) / (x1-x2)
            slopes.append(slope)
            c = y1 - slope * x1
            consts.append(c)
    avg_slope = cal_avg(slopes)
    avg_consts = cal_avg(consts)
    
    # Calculate average intersection at lower_border.
    if avg_slope == 0: return None
    x_lane_lower_point = int((lower_border - avg_consts) / (avg_slope))
    
    # Calculate average intersection at upper_border.
    x_lane_upper_point = int((upper_border - avg_consts) / (avg_slope))
    
    return [x_lane_lower_point, lower_border, x_lane_upper_point, upper_border]
def distance(x1, y1, x2, y2, px, py):
    
    # Calculate the distance between the point and the vector
    dx = x2 - x1
    dy = y2 - y1
    numerator = abs(dy * px - dx * py + x2 * y1 - y2 * x1)
    denominator = math.sqrt(dx ** 2 + dy ** 2)
    if denominator == 0: distance = -1
    distance = numerator / denominator

    # Calculate the projection of the point onto the vector
    t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / ((x2 - x1) ** 2 + (y2 - y1) ** 2)
    proj_x = int(x1 + t * (x2 - x1))
    proj_y = int(y1 + t * (y2 - y1))

    return distance, (proj_x, proj_y)
def make_way(id,img,roi_vertices,threshold,minLineLength,lanes_img,center_point):
    midline_x1 = None
    midline_x2 = None
    lane_left,lane_right=None,None
    ratio = 3.18904
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    threshold_img = cv2.inRange(gray, threshold[0], threshold[1])
    # Defining a blank mask.
    mask = np.zeros_like(threshold_img)
    
    # Defining a 3 channel or 1 channel color to fill the mask.
    if len(threshold_img.shape) > 2:
        channel_count = threshold_img.shape[2]  # 3 or 4 depending on the image.
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # Filling pixels inside the polygon.
    cv2.fillPoly(mask, roi_vertices, ignore_mask_color)

    # Constructing the region of interest based on where mask pixels are nonzero.
    roi = cv2.bitwise_and(threshold_img, mask)
    # Smooth with a Gaussian blur.
    roi_blur = cv2.GaussianBlur(roi, (3,3), 0)

    # Perform Edge Detection.
    canny_blur = cv2.Canny(roi_blur, threshold[0], threshold[1])
    cv2.imshow('canny'+id,canny_blur)
    
    lines = cv2.HoughLinesP(canny_blur, 1, np.pi/180, 90, np.array([]), minLineLength=minLineLength, maxLineGap=50)

    # Draw all lines found onto a new image.
    hough = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
    draw_lines(hough, lines)

    # Define bounds of the region of interest.
    roi_upper_border = roi_vertices[0][3][1]
    roi_lower_border = roi_vertices[0][1][1]

    # Create a blank array to contain the (colorized) results.

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
    x_mid_line = None
    if lane_left is not None and lane_right is not None:# and not np.isnan(lane_left).any() and not np.isnan(lane_right).any():
        midline_x1=int((x1_left + x1_right)/2)
        midline_x2=int((x2_left + x2_right)/2)

        cv2.line(lanes_img,(midline_x1,y1_left),(midline_x2,y2_left),(0,255,0),10)
        rldis = abs(x2_left - x2_right) * ratio
        cv2.putText(lanes_img, "Lane distance: " + str(round(rldis,2)) + ' mm', (50,150), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color= (255, 255, 255), lineType=2)
    
    center_point = (int(img.shape[1]/2), int(img.shape[0]/2))    
    if midline_x1 is not None:
        dis, proj_point = distance(midline_x1, y1_left, midline_x2, y2_left, center_point[0], center_point[1])
        cv2.circle(lanes_img,center_point,5,(200,0,0),10)
        # cv2.line(lanes_img,center_point, (midline_x2,y2_left),(0,255,255),10)
        # cv2.line(lanes_img,center_point, (center_point[0],height),(0,255,255),10)
        dis_mm = dis / ratio
        if center_point[0] > proj_point[0]: 
            cv2.putText(lanes_img, "Left: " + str(round(dis_mm,2)) + ' mm', (50,100), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color= (255, 255, 255), lineType=2)
        else:
            cv2.putText(lanes_img, "Right: "+ str(round(dis_mm,2)) + 'mm', (50,100), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color= (255, 255, 255), lineType=2)

    image_annotated = cv2.addWeighted(img, 0.8, lanes_img, 1, 0)
    return image_annotated,lane_left,lane_right,midline_x1
def calculate_ratio(warp):
    CHESSBOARD_CORNER_NUM_X = 9
    CHESSBOARD_CORNER_NUM_Y = 6
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_CORNER_NUM_X,CHESSBOARD_CORNER_NUM_Y), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        #Draw and display the corners
        cv2.drawChessboardCorners(warp, (CHESSBOARD_CORNER_NUM_X,CHESSBOARD_CORNER_NUM_Y), corners2, ret)
    x1, y1 = corners2[0][0]
    x2 = corners2[53][0]
    y2 = corners2[8][0]
    y_square_pixel = cv2.norm(y1 - y2)/10

    x_square_pixel = cv2.norm(x1 - x2)/10
    square_mm = 22.28 #mm
    x_pixel_per_mm = x_square_pixel / square_mm
    y_pixel_per_mm = y_square_pixel / square_mm

    return (x_pixel_per_mm, y_pixel_per_mm)
def calculate_cam_distance(image):
    return
def nextMove(imshape,leftLines,rightLines, x_mid_line):
    goodinterval = [imshape[1] * 2/5, imshape[1] * 3/5]
    rightSpeed = 0
    leftSpeed = 0
    if leftLines is None and rightLines is None:
        rightSpeed = 10
        leftSpeed = 25
    elif leftLines is None:
        rightSpeed = 25
        leftSpeed = 10
    elif  rightLines is None:
        rightSpeed = 10
        leftSpeed = 35
    elif x_mid_line is None:
        rightSpeed = 10
        leftSpeed = 35
    elif x_mid_line < goodinterval[0]:
        rightSpeed = 35
        leftSpeed = 10
    elif x_mid_line > goodinterval[1]:
        rightSpeed = 10
        leftSpeed = 40
    elif goodinterval[0] < x_mid_line <goodinterval[1]:
        rightSpeed = 50
        leftSpeed = 50

    return leftSpeed, rightSpeed
