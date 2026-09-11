import sys
sys.path.insert(0, ".")
import src
import cv2
import numpy as np
from PIL import Image

def detect_paper_margins(pil_img):
    gray = np.array(pil_img.convert("L"))
    h, w = gray.shape
    
    # Analyze row brightness profiles
    row_means = np.mean(gray, axis=1)
    col_means = np.mean(gray, axis=0)
    
    paper_median = np.median(gray[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)])
    thresh = paper_median * 0.55
    print(f"Paper median: {paper_median:.1f}, background threshold: {thresh:.1f}")
    
    # Top margin
    top = 0
    while top < h * 0.2 and row_means[top] < thresh:
        top += 1
        
    # Bottom margin
    bottom = h - 1
    while bottom > h * 0.8 and row_means[bottom] < thresh:
        bottom -= 1
        
    # Left margin
    left = 0
    while left < w * 0.2 and col_means[left] < thresh:
        left += 1
        
    # Right margin
    right = w - 1
    while right > w * 0.8 and col_means[right] < thresh:
        right -= 1
        
    print(f"Original size: ({w}, {h}) -> Cropped bbox: (left={left}, top={top}, right={right}, bottom={bottom})")
    cropped = pil_img.crop((left, top, right + 1, bottom + 1))
    return cropped

img = Image.open("doc2.jpeg")
detect_paper_margins(img)
