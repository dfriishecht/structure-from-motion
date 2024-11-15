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
    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key = lambda x:x.distance)
    return matches

def calcualte_essential_matrix(K, F):
    E = None
    
    # E = KT * F * K

    return E

def get_epipolar_line(pos1, pos2, F):
    L1, L2 = None, None
    # Epipolar restriction
    # pos2T * F * pos1 = 0

    # compute left and right lines
    return L1, L2

def get_camera_pose():
    pass


def draw_matches(img1, img2, kp1, kp2, matches, num_matches=None):
    if not num_matches:
        num_matches = len(matches)
        
    img_matched = cv.drawMatches(img1,kp1,img2,kp2,matches[:num_matches],None) # type: ignore
    return img_matched
