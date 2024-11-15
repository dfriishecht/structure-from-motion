import numpy as np
import cv2 as cv
import os
from pathlib import Path

Path('capture/').mkdir(parents=True, exist_ok=True)
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera!")
    exit()
i = 0
while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame. Exiting...")
        break
    
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('c'):
        cv.imwrite(f'capture/frame{i}.png', frame)
        i += 1
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows
