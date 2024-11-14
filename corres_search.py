"""
This file will contain helper functions for the `Correspondence Search` part of the SFM pipeline.
- Feature extraction
- Matching
- Geometric verification
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def extract_features(img):
    sift = cv.SIFT.create()
    kp, desc = sift.detectAndCompute(img, None) # type: ignore
    return kp,desc

def match_keypoints():
    pass

def calcualte_essential_matrix():
    pass

def get_camera_pose():
    pass


if __name__ == "__main__":
    img = cv.imread("data/fountain-P11/images/0004.jpg")
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    kp, desc = extract_features(img)
    
    img=cv.drawKeypoints(img ,
                      kp ,
                      img ,
                      flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    

    fig,ax=plt.subplots(ncols=1,figsize=(9,4)) 
    ax.imshow(img)
    plt.show()

