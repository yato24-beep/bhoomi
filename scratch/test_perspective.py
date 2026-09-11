import sys
sys.path.insert(0, ".")
import src
import cv2
import numpy as np
from PIL import Image

def test_rectify(image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    orig_area = w * h
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    best_quad = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.25 * orig_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            best_quad = approx.reshape(4, 2)
            break
            
    print("Found 4-point quad:", best_quad is not None)
    if best_quad is not None:
        print("Quad coordinates:\n", best_quad)

test_rectify("doc2.jpeg")
