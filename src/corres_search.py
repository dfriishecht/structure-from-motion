"""
This file will contain helper functions for the `Correspondence Search` part of the SFM pipeline.
- Feature extraction
- Matching
- Geometric verification
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def extract_features(img:np.ndarray, intr_mat=None, dist_coeff=None) -> tuple:
    """
    Extract SIFT feature points from input image; undistort the image if camera parameters are known

    Args:
        img
        intr_mat
        dist_coeff

    """
    if intr_mat and dist_coeff:
        img = cv.undistort(img, intr_mat, dist_coeff) # type: ignore

    sift = cv.SIFT.create()
    kp, desc = sift.detectAndCompute(img, None) # type: ignore
    return kp,desc

def match_keypoints(desc1, desc2):
    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key = lambda x:x.distance)
    return matches

def calculate_fund_matrix(pts1, pts2):
    """
    Calculate the fundamental matrix of two images.
    This matrix describes the epipolar geometry between two images

    Args: 
        pts1: matched points from image 1
        pts2: matched points from image 2

    Returns: 
        F: Fundamental matrix
        mask: mask indicating which point belong to inliers/outliers (1/0)
    """

    F,mask = cv.findFundamentalMat(pts1, pts2, cv.RANSAC)
    return F, mask


def estimate_essential_matrix(K, pts1, pts2):
    E = None
    
    

    return E

def get_epipolar_line(pts1, pts2, F):
    L1, L2 = None, None
    # Epipolar restriction
    # pts2T * F * pts1 = 0

    # compute left and right lines
    return L1, L2

def get_camera_pose():
    pass


def draw_matches(img1, img2, kp1, kp2, matches, num_matches=None):
    if not num_matches:
        num_matches = len(matches)
        
    img_matched = cv.drawMatches(img1,kp1,img2,kp2,matches[:num_matches],None) # type: ignore
    return img_matched
