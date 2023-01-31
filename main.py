import math

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

filename = 'data/37e45e-20220116144603_1.jpg'
img = cv.imread(filename)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_p = img.copy()
img = cv.GaussianBlur(img, (3, 3), 0)

# hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(29, 29))
img = clahe.apply(img)

cv.imwrite('data/temp/temp.jpg', img)
img = cv.imread('data/temp/temp.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
# hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)
# kernel = np.ones((3, 3), np.uint8)
#
# lower_white = np.array([0, 210, 10])
# upper_white = np.array([100, 255, 50])
#
# lower_yellow = np.array([0, 80, 0])
# upper_yellow = np.array([25, 100, 25])
#
# mask1 = cv.inRange(hls, lower_yellow, upper_yellow)
# mask2 = cv.inRange(hls, lower_white, upper_white)
#
# mask_it = cv.bitwise_or(mask2, mask1, mask=None)
#
# res = cv.bitwise_and(img, img, mask=mask_it)

# cv.imwrite('data/temp/temp.jpg', res)
# img = cv.imread('data/temp/temp.jpg', 0)

# img = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
dst = cv.Canny(thresh, 50, 200, None, 3)
#
# opening = cv.morphologyEx(dst, cv.MORPH_OPEN, kernel)
lines_p = cv.HoughLinesP(dst, rho=1, theta=np.pi / 180, threshold=25, minLineLength=10, maxLineGap=30)

for i in range(len(lines_p)):
    x_1, y_1, x_2, y_2 = lines_p[i][0]
    cv.line(img_p, (x_1, y_1), (x_2, y_2), (0, 255, 0), 4)

print("code successful!")
cv.imshow("Hough_line_p", img_p)

cv.imshow("Source", dst)
# cv.imshow("res", dst)
cv.waitKey()

# plt.imshow(res)
# plt.show()
