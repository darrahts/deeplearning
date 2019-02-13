# -*- coding: utf-8 -*-

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

def canny(image_color):
            
    threshold_1 = 20
    threshold_2 = 100
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    erode = cv2.erode(image_gray, None, iterations=2)
    dialated = cv2.dilate(erode, None, iterations=2)
      
    canny = cv2.Canny(dialated, threshold_1, threshold_2)
    return canny


def Laplace(image_color):

    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
    return laplacian
    
def sobely(image_color):

    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    y_sobel = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize = 7)
    return y_sobel
    
def sobelx(image_color):

    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    x_sobel = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize = 7)
    return x_sobel
    
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() # Cap.read() returns a ret bool to indicate success.
    cv2.imshow('Live Edge Detection', canny(frame))
    cv2.imshow('Webcam Video', frame)
    if cv2.waitKey(1) == 13: #13 Enter Key
        break
    
#while True:
#    ret, frame = cap.read() # Cap.read() returns a ret bool to indicate success.
#    cv2.imshow('Live Edge Detection', Laplace(frame))
#    cv2.imshow('Webcam Video', frame)
#    if cv2.waitKey(1) == 13: #13 Enter Key
#        break
#    
#while True:
#    ret, frame = cap.read() # Cap.read() returns a ret bool to indicate success.
#    cv2.imshow('Live Edge Detection', sobely(frame))
#    cv2.imshow('Webcam Video', frame)
#    if cv2.waitKey(1) == 13: #13 Enter Key
#        break
#    
#while True:
#    ret, frame = cap.read() # Cap.read() returns a ret bool to indicate success.
#    cv2.imshow('Live Edge Detection', sobelx(frame))
#    cv2.imshow('Webcam Video', frame)
#    if cv2.waitKey(1) == 13: #13 Enter Key
#        break
        
cap.release() # camera release 
cv2.destroyAllWindows() 