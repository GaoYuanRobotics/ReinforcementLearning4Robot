#! /usr/bin/env python  
# START--a workaround to import both cv2 and rospy--
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import rospy
# END--a workaround to import both cv2 and rospy--

import os
import time
import math
# commented on 2018.10.2 5:21PM by LXC
#from detect.msg import ball_detect
#from detect.msg import ball_location

def detect_ball():

    Image = cv2.imread('/home/lu/pycode_lxc/image/image_big.png')
    Gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    #B,G,R = cv2.split(Image)
    rows, cols, channels = Image.shape
    diff = 30
    min_row = rows
    max_row = 0
    min_col = cols
    max_col = 0
    center = (0, 0)
    for row in range(rows):
        for col in range(cols):
            if ((Image[row, col, 2] > Image[row, col, 0]) and (Image[row, col, 2] - Image[row, col, 0] > diff)):
                if ((Image[row, col, 2] > Image[row, col, 1]) and (Image[row, col, 2] - Image[row, col, 1] > diff)):
                    Gray[row, col] = 255
                    if (min_row > row):
                        min_row = row
                    if (max_row < row): 
                        max_row = row
                    if (min_col > col):
                        min_col = col
                    if (max_col < col):
                        max_col = col
                else:
                    Gray[row, col] = 0
            else:
                Gray[row, col] = 0

    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(7, 7))
    Gray = cv2.erode(Gray,kernel1)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(7, 7))
    Gray = cv2.dilate(Gray,kernel2)
    ret, thresh = cv2.threshold(Gray, 0, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    Len = len(contours[1])
    w1 = 0
    w2 = 0
    if (Len):
        for k in range(len(contours[1])):
            if (len(contours[1][k]) >= w1):
                w1 = len(contours[1][k])
                w2 = k

        (x, y), radius = cv2.minEnclosingCircle(contours[1][w2])
        center = (int(x), int(y))
        print (center, int(radius))
        cv2.circle(Image, center, int(radius), (0, 0, 255), -1)
        cv2.circle(Image, center, 3, (0, 255, 255), -1) 
        cv2.imwrite('/home/lu/pycode_lxc/image/res1.png',Image)
        cv2.imwrite('/home/lu/pycode_lxc/image/thresh1.png',thresh)
    return center


detect=detect_ball()
