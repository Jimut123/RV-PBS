import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt



img = cv.imread('IMG_2799.jpg')

print(img.shape)
img = cv.resize(img,(400, 300), cv.INTER_AREA)

# Convert BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# define range of blue color in HSV
lower_blue = np.array([100,20,20])
upper_blue = np.array([300,255,255])
# Threshold the HSV image to get only blue colors
mask = cv.inRange(hsv, lower_blue, upper_blue)
# Bitwise-AND mask and original image
res = cv.bitwise_and(hsv,hsv, mask= mask)

plt.figure(figsize=(20, 10))
plt.subplot(121)
plt.imshow(img)

plt.subplot(122)
plt.imshow(mask)



# noise removal
kernel = np.ones((5,5),np.uint8)

opening = cv.morphologyEx(mask,cv.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=1)

plt.figure(figsize=(10,10))
plt.subplot(221)
plt.imshow(img)
plt.subplot(222)
plt.imshow(mask)
plt.subplot(223)
plt.imshow(opening)
plt.subplot(224)
plt.imshow(sure_bg)
plt.show()
