# -*- coding: utf-8 -*-

#convolution use a kernel matrix to scan an image and apply a filter
#convolution preserves the spatial relationship between pixels

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

image2 = cv2.imread('dtbg.jpg')
image2.shape

b,g,r = cv2.split(image2)

image2[:,:,2] = b
image2[:,:,0] = r

plt.imshow(image2)
cv2.imshow('us', image2)
cv2.waitKey()
cv2.destroyAllWindows()

image2[:,:,2] = r
image2[:,:,0] = b

gray_img = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img)

cv2.imshow('us', gray_img)
cv2.waitKey()
cv2.destroyAllWindows()

# Sharpening Kernel #1
Sharp_Kernel = np.array([[0,-1,0], 
                        [-1,5,-1], 
                        [0,-1,0]])

# Sharpening Kernel #2
Sharp_Kernel = np.array([[-1,-1,-1], 
                         [-1,9,-1], 
                         [-1,-1,-1]])


Sharpened_Image = cv2.filter2D(image2, -1, Sharp_Kernel)
cv2.imshow('Sharpened Image', Sharpened_Image)

cv2.waitKey(0)
cv2.destroyAllWindows()


Blurr_Kernel = np.ones((9,9))*1/64
Blurred_Image = cv2.filter2D(image2, -1, Blurr_Kernel)

cv2.imshow('Blurred Image', Blurred_Image)

cv2.waitKey(0)
cv2.destroyAllWindows()


















