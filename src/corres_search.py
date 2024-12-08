"""
This file will contain helper functions for the `Correspondence Search` part of the SFM pipeline.
- Feature extraction
- Matching
- Geometric verification
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def extract_features(img: np.ndarray, intr_mat=None, dist_coeff=None) -> tuple:
    """
    Extract SIFT feature points from input image; undistort the image if camera parameters are known

    Args:
        img
        intr_mat
        dist_coeff

    """
    if intr_mat and dist_coeff:
        img = cv.undistort(img, intr_mat, dist_coeff)  # type: ignore

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT.create()
    kp, desc = sift.detectAndCompute(img, None)  # type: ignore
    return kp, desc


def match_keypoints_bf(img0, img1):
    kp0, desc0 = extract_features(img0)
    kp1, desc1 = extract_features(img1)
    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=False)
    matches = bf.knnMatch(desc0, desc1, k= 2)

    good = []
    for m,n in matches:
        if m.distance < 0.70 * n.distance:
            good.append(m)
        
    pts0 = np.float32([kp0[m.queryIdx].pt for m in good]) # type: ignore
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good]) # type: ignore
    return pts0, pts1


def match_keypoints_flann(
    img1, img2, n_trees=5, n_checks=50, lowe_ratio=0.8
):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(
        algorithm=FLANN_INDEX_KDTREE, trees=n_trees
    )  # number of trees in the KD-Tree, higher the better, but slower
    search_params = dict(
        checks=n_checks
    )  # number of recursive checks, higher the better, but slower
    flann = cv.FlannBasedMatcher(index_params, search_params)  # type: ignore

    kp1, desc1 = extract_features(img1)
    kp2, desc2 = extract_features(img2)

    matches = flann.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for m, n in matches:
        if m.trainIdx < len(kp2) and n.trainIdx < len(kp2):
            if m.distance < lowe_ratio * n.distance:
                good_matches.append(m)

    left_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    right_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    return left_pts, right_pts


def normalize_points(points):
    """
    Normalize 2D points for the 8-point algorithm.

    1. Shifting the points so their centroid is at the origin.
    2. Scaling the points so that the average distance from the origin is sqrt(2).

    Args:
        points: Input 2D points, (N, 2), where N is the number of points.

    Returns:
        tuple:
            - norm_points: Normalized 2D points, shape (N, 2).
            - T: Transformation matrix of shape (3, 3) that was used for normalization.
    """
    # Averaging the x points and y points, necessary
    centroid = np.mean(points, axis=0)

    # Shift the points to make the centroid the origin
    shifted_points = points - centroid

    # Calculate the average distance from the origin
    avg_distance = np.mean(np.sqrt(np.sum(shifted_points**2, axis=1)))

    # Making the average distance sqrt(2) for nicer calculations
    # This is to have an average difference of sqrt(2) from origin in both images
    scale = np.sqrt(2) / avg_distance

    # Transformation Matrix for homogenous points
    # homogenous points multiplied by T gives you normalized points
    T = np.array(
        [[scale, 0, -scale * centroid[0]], [0, scale, -scale * centroid[1]], [0, 0, 1]]
    )

    # Normalize the points
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack([points, ones])  # Convert to homogeneous coordinates
    norm_points_homogeneous = (T @ points_homogeneous.T).T
    norm_points = norm_points_homogeneous[
        :, :2
    ]  # Convert back to Cartesian coordinates

    return norm_points, T


def calculate_fund_matrix(pts1, pts2):
    """
    Calculate the fundamental matrix using the 8-point algorithm.

    Args:
        pts1 : Points from image 1, shape (N, 2).
        pts2 : Points from image 2, shape (N, 2).

    Returns:
        F : Fundamental matrix, shape (3, 3).
    """

    # Normalize the points
    norm_pts1, T1 = normalize_points(pts1)
    norm_pts2, T2 = normalize_points(pts2)

    # Making matrix A
    # A^T * Fx = 0
    A = []
    for (x1, y1), (x2, y2) in zip(norm_pts1, norm_pts2):
        A.append(
            [x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1]
        )  # 8 point algorithm matrix
    A = np.array(A)

    # Solve Af = 0 using SVD
    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)  # Getting the fundamental matrix (last row of Vt)

    # Enforce rank 2, mapping the 2D points in 2 images, so it is necessary, not 3D points
    U, S, Vt = np.linalg.svd(F)
    S = np.diag(S)
    S[2, 2] = 0
    F = U @ S @ Vt

    # Denormalize the fundamental matrix to original points
    F = T2.T @ F @ T1

    return F


def calculate_fund_mat_ransac(pts1, pts2):
    pass


def estimate_essential_matrix(K, pts1, pts2):
    # E = K2.T @ F @ K1
    E, mask = cv.findEssentialMat(pts1, pts2, K, method=cv.RANSAC)
    return E, mask


def get_epipolar_line(img1, img2, F, pts1, pts2):
    """
    Draw epipolar lines on the images given the fundamental matrix and point correspondences.

    Args:
        img1: First image.
        img2: Second image.
        F: Fundamental matrix.
        pts1: Points from the first image, shape (N, 2).
        pts2: Points from the second image, shape (N, 2).

    Returns:
        img1_with_lines: Image 1 with epipolar lines and points drawn.
        img2_with_lines: Image 2 with epipolar lines and points drawn.
    """

    # Convert points to homogeneous points
    pts1_hom = np.hstack(
        [pts1, np.ones((pts1.shape[0], 1))]
    )  # Just adding one at the end
    pts2_hom = np.hstack([pts2, np.ones((pts2.shape[0], 1))])

    # Equations for epipolar lines
    # lines for img 1 = F transpose * img2
    # lines for img 2 = F * img 1
    lines1 = (F.T @ pts2_hom.T).T
    lines2 = (F @ pts1_hom.T).T

    # Reference: OpenCV Epipole Geometry
    # VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
    # Get image dimensions
    h, w = img1.shape[:2]

    # Draw lines and points on both images
    for r, pt1, pt2 in zip(lines1, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())

        # Line in img1
        x0, y0 = map(int, [0, -r[2] / r[1]]) if r[1] != 0 else (0, 0)
        x1, y1 = map(int, [w, -(r[2] + r[0] * w) / r[1]]) if r[1] != 0 else (w, h)
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv.circle(img1, tuple(map(int, pt1)), 5, color, -1)

        # Line in img2
        r2 = lines2[
            np.where((lines1 == r).all(axis=1))[0][0]
        ]  # Corresponding line in img2
        x0, y0 = map(int, [0, -r2[2] / r2[1]]) if r2[1] != 0 else (0, 0)
        x1, y1 = map(int, [w, -(r2[2] + r2[0] * w) / r2[1]]) if r2[1] != 0 else (w, h)
        img2 = cv.line(img2, (x0, y0), (x1, y1), color, 1)
        img2 = cv.circle(img2, tuple(map(int, pt2)), 5, color, -1)

    return img1, img2


def get_camera_pose():
    pass


def draw_matches(img1, img2, kp1, kp2, matches, num_matches=None):
    if not num_matches:
        num_matches = len(matches)

    img_matched = cv.drawMatches(img1, kp1, img2, kp2, matches[:num_matches], None)  # type: ignore
    return img_matched
