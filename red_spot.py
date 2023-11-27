import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read the Image
image_path = '/content/image8.jpg'
image = cv2.imread(image_path)

# Step 2: Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Step 3: Define the red color range in HSV
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 245, 255])
red_mask1 = cv2.inRange(hsv, lower_red, upper_red)

lower_red = np.array([160, 100, 100])
upper_red = np.array([180, 255, 255])
red_mask2 = cv2.inRange(hsv, lower_red, upper_red)

red_mask = cv2.bitwise_or(red_mask1, red_mask2)

# Step 4: Noise reduction using morphological operations
kernel = np.ones((5, 5), np.uint8)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

# Step 5: Contour detection
contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 6: Filter and draw contours
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 100:  # Adjust this threshold based on your needs
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

# Display the result using Matplotlib
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
