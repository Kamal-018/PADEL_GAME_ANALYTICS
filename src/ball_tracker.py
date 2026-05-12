# ball_tracker.py
import cv2
import numpy as np


LOWER_RED1 = np.array([0, 150, 150])
UPPER_RED1 = np.array([5, 255, 255])
LOWER_RED2 = np.array([175, 150, 150])
UPPER_RED2 = np.array([180, 255, 255])


COURT_PTS = np.array([
    [663, 93],
    [1234, 75],
    [1833, 944],
    [128, 986]
], dtype=np.int32)

def detect_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    mask2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    mask = cv2.bitwise_or(mask1, mask2)

   
    poly_mask = np.zeros_like(mask)
    cv2.fillPoly(poly_mask, [COURT_PTS], 255)
    mask = cv2.bitwise_and(mask, poly_mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        valid = [c for c in contours if 50 < cv2.contourArea(c) < 300]
        if valid:
            c = max(valid, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cx = x + w // 2
            cy = y + h // 2
            return (cx, cy, x, y, w, h)  

    return None