
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gs
import numpy as np


# ### general functions

# In[2]:


def ROI(img):
    """returns the region of interest mask for an image"""
    h = img.shape[0]
    #here are the points set for the polygon, since there are 4 points this is a 4 sided poly
    trapezoid = np.array([[(200,h), (1100,h), (750, 300), (500, 300)]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, trapezoid, (255, 255,255))
    return mask

def RGBImage(img):
    """
    used for displaying images in matplotlib that are read with cv2. cv2 colorspace is bgr and normal is rgb.
    this works fine on my desktop but doesnt work on my laptop (due to version issues of course)
    """
    try:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        return img
    
def Canny(img):
    """
    Converts an image to gray scale, applies a gaussian blur (to reduce image noise),
    then dilates the image to increase the line size / weight, and finally returns the canny.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9), 0)
    can = cv2.Canny(blur, 40, 120)
    ero = cv2.dilate(can, (11,11), iterations=5)
    return ero

def maskLines(img, lines):
    """
    returns a mask of the same size as <@param img>, with the <@param lines> found
    TODO: add the hough transform code here, not in the main
    """
    lineImg = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(lineImg, (x1, y1), (x2, y2), (0,255,0), 10)
    return lineImg

def avgMB(img, lines):
    """
    smoothes the different <@param lines> using np.polyfit
    """
    leftFit = []
    rightFit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        m,b = np.polyfit((x1,x2), (y1,y2), 1)
        if(m < 0):
            leftFit.append((m, b))
        else:
            rightFit.append((m,b))
    
    avgLeftMB = np.average(leftFit, axis=0)
    avgRightMB = np.average(rightFit, axis=0)
    leftLine = unpackLine(img, avgLeftMB)
    rightLine = unpackLine(img, avgRightMB)
    return np.array([leftLine, rightLine])

def unpackLine(img, lineParams):
    """
    returns x1,y1,x2,y2 for a line
    """
    m,b = lineParams
    #the start of the lines is the bottom of the image (i.e. its height)
    y1 = img.shape[0]
    #the end of the lines will be 1/2 the way up the image, play with this value!
    y2 = int(y1*.5)
    x1 = int((y1 - b) / m)
    x2 = int((y2 - b) / m)
    return np.array([x1,y1,x2,y2])
        



cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    ret, frame = cap.read()
    cny = Canny(frame)
    msk = ROI(cny)
    mskCny = cv2.bitwise_and(cny, msk)
    lines = cv2.HoughLinesP(mskCny, 9, np.pi/180, 100, np.array([]), minLineLength=100, maxLineGap=15)
    try:
        avgLines = avgMB(frame, lines)
        maskLinesAvg = maskLines(frame, avgLines)
        result = cv2.addWeighted(frame, 1.0, maskLinesAvg, .8, 1)
        cv2.imshow('res', result)
    except Exception as e: #i.e. when there is nothing else in the stream to read
        print("dd")
    if(cv2.waitKey(1) == ord('q')):
       break
cap.release()
cv2.destroyAllWindows()
print("done")
