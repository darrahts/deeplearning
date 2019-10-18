# -*- coding: utf-8 -*-

## SOBEL NOTE (first order derivative) peak crossing
# the orientation of the edge is the arctan(Gy/Gx)

#where Gx = np.array([[1,0,-1], [2,0,-2], [1,0,-1]]) 
#      Gy = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])

#strength of the edge is sqrt(Gx^2 + Gy^2)

## LAPLACIAN NOTE (second order derivative) zero order crossing
# sensitive to noise

## CANNY EDGE NOTE 
# smooth the image with a guassian filter
# compute gradient magnitude and direction
# thinning zero out all pixels that are not the max along the dir of gradient
# thresholding removes noise


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

Gx = np.array([[1,0,-1], [2,0,-2], [1,0,-1]])
Gx

image2 = cv2.imread('dtbg.jpg')
image2.shape

gray_img = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


x_sobel = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize = 7)
cv2.imshow('Sobel - X direction', x_sobel)
cv2.waitKey()
cv2.destroyAllWindows()

y_sobel = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize = 7)
cv2.imshow('Sobel - Y direction', y_sobel)
cv2.waitKey()
cv2.destroyAllWindows()

laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
cv2.imshow('Laplacian', laplacian)
cv2.waitKey()
cv2.destroyAllWindows()



canny = cv2.Canny(gray_img, 100,150)

cv2.imshow('Canny', canny)
cv2.waitKey()
cv2.destroyAllWindows()

canny = cv2.Canny(gray_img, 150, 200)

cv2.imshow('Canny', canny)
cv2.waitKey()
cv2.destroyAllWindows()

canny = cv2.Canny(gray_img, 200, 255)

cv2.imshow('Canny', canny)
cv2.waitKey()
cv2.destroyAllWindows()




