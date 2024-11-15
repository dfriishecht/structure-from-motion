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

def match_keypoints(desc1, desc2):
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key = lambda x:x.distance)
    return matches

def draw_matches(img1, img2, kp1, kp2, matches, num_matches):
    img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:num_matches],None,flags=cv.DrawMatchesFlags_NOT_DRAWN_SINGLE_POINTS)
    return img3

def calcualte_essential_matrix():
    pass

def get_camera_pose():
    pass
