import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
sys.path.insert(0, r"c:\Land Record")
import src
from PIL import Image
import numpy as np
import cv2

rot90 = Image.open(r"c:\Land Record\scratch\new_kothi_rot90.png")
gray = cv2.cvtColor(np.array(rot90), cv2.COLOR_RGB2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
print("Connected components:", num_labels)
for i in range(1, num_labels):
    x, y, w, h, area = stats[i]
    print(f"Component {i}: x={x}, y={y}, w={w}, h={h}, area={area}")
    crop = rot90.crop((x, y, x+w, y+h))
    crop.save(rf"c:\Land Record\scratch\comp_{i}.png")
