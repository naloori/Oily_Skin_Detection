import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('image1.png')
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
lower_bound = (100, 0, 0)  # Lower bound of red color
upper_bound = (255, 250, 0)  # Upper bound of red color
mask = cv2.inRange(rgb, lower_bound, upper_bound)
result = cv2.bitwise_and(rgb, rgb, mask=mask)
plt.ion()
plt.imshow(result)
plt.show(block=True)
