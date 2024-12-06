"""
This file will contain helper functions for the `Incremental Reconstruction` part of the SFM pipeline.
- Initialization
- Image Registration
- Triangulation
- Bundle Adjustment
- Outlier Filtering
- Plotting
"""
import numpy as np
import cv2 as cv


def triangulate(P0: np.ndarray, P1: np.ndarray, pts0: np.ndarray, pts1: np.ndarray):
    """
    Triangulate two sets of 2D points into one set of 3D coordinates

    Args:
        P0: Projection matrix for first image
        P1: Projection matrix for second image
        pts0: 2D points from first image
        pts1: 2D points from second image

    Returns:
        pts_3d: 3D points
    """

    def get_A_mat(uv0, uv1):
        u0,v0 = uv0[0], uv0[1]
        u1, v1 = uv1[0], uv1[1]

        A = np.array([
                    u0 * P0[2] - P0[0],
                    v0 * P0[2] - P0[1],
                    u1 * P1[2] - P1[0],
                    v1 * P1[2] - P1[1]
        ])

        return A
    
    points_3d = []
    # Solve AX = 0 for each feature point pair
    for uv0, uv1 in zip(pts0, pts1):
        A_mat = get_A_mat(uv0, uv1)
        _, _, V = np.linalg.svd(A_mat)
        X_hom = V[-1]
        X_cartesian = X_hom[0:3] / X_hom[3]
        points_3d.append(X_cartesian)

    return points_3d


def PnP(X, p, K, d, p_0, initial):
    """
    Recover camera rotation and translation from 3D and 2D points

    Args:
        X: 3D points
        p: Corresponding 2D points
        K: Camera intrinsic matrix
        d: distortion coefficient
        p_0: 

    """
    if initial == 1:
        X = X[:, 0, :]
        p = p.T
        p_0 = p_0.T


    ret, rvecs, t, inliers = cv.solvePnPRansac(X, p, K, d, cv.SOLVEPNP_ITERATIVE) # type: ignore

    R, _ = cv.Rodrigues(rvecs)


    # Filter out bad 2D-3D correspondences
    if inliers is not None:
        p = p[inliers[:, 0]]
        X = X[inliers[:, 0]]
        p_0 = p_0[inliers[:, 0]]

    return R, t, p, X, p_0




if __name__ == "__main__":

    P0 = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0]
    ])

    P1 = np.array([
        [1, 0, 0, -1],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])

    # pts0 = np.random.rand(10,1)
    # pts1 = np.random.rand(10,1)

    pts0 = np.array([[1, 2], [0.5, 1.5], [1.2, 1.8]])
    pts1 = np.array([[0.9, 1.8], [0.4, 1.4], [1.1, 1.9]])

    print(pts0.T.shape)

    triangulate(P0, P1, pts0, pts1)
    point3d = (cv.triangulatePoints(P0, P1, pts0.T, pts1.T))
    print(point3d)

