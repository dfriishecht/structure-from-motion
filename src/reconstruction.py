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
from scipy.optimize import least_squares


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

    return np.array(points_3d)

# Calculation for Reprojection error in main pipeline
def reprojection_error(X, pts, Rt, K, homogenity):
    """
    Calculates the error between 2D image points and the corresponding 3D
    points projected back down to the image plane.

    Args:
        X: 4xN matrix of 3D points in homogenous coordinates.
        pts: Nx2 array of 2D image points that correspond to 3D points in X.
        Rt: A 3x4 matrix representing the camera pose (rotation R and translation t).
        K: The 3x3 camera intrinsic matrix
        homogenity: A flag for if the 3D points are homogenous (1) or cartesian (0).
    
    Returns:
        total_error
        X
        proj
    """
    total_error = 0
    R = Rt[:3, :3]
    translation = Rt[:3, 3]

    rotation, _ = cv.Rodrigues(R)

    if homogenity:
       X = cv.convertPointsFromHomogeneous(X.T)
    
    proj, _ = cv.projectPoints(X, rotation, translation, K, distCoeffs=None) #proj is Nx2 array of reprojected 2D points
    proj = np.float32(proj[:, 0, :])
    pts = np.float32(pts)
    if homogenity:
        total_error = cv.norm(proj, pts.T, cv.NORM_L2)
    else:
        total_error = cv.norm(proj, pts, cv.NORM_L2)
    pts = pts.T
    total_error = total_error / len(proj)

    return total_error, X, proj


def optimal_reprojection(x):
    """
    Calculates the reprojection error for a given set of paramters.
    
    Args: 
        x : A 1D array containing extrinsic, intrinsic,
                        2D points, and 3D points in sequence.
    Returns: 
    error_array
    """
    Rt = x[0:12].reshape((3, 4))
    K = x[12:21].reshape((3, 3))
    remaining_points = len(x[21:])
    num_2D_points = int(remaining_points * 0.4)  # 40% 2D, 60% 3D
    image_points = x[21:21 + num_2D_points].reshape((2, num_2D_points // 2))
    X = x[21 + num_2D_points:].reshape((int(len(x[21 + num_2D_points:]) / 3), 3))
    R = Rt[:3, :3]
    t = Rt[:3, 3]
    
    # Convert rotation matrix to Rodrigues vector
    rotation_vector, _ = cv.Rodrigues(R)

    # Project 3D points into 2D
    points_2D, _ = cv.projectPoints(X, rotation_vector, t, K, distCoeffs=None)
    points_2D = points_2D[:, 0, :]  #  (N, 2) for 2D

    
    errors = []
    image_points = image_points.T  # Convert to shape (N, 2)
    num_points = len(image_points)

    for index in range(num_points):
        observed = image_points[index] # existing points
        reprojected = points_2D[index] # new points
        error = (observed - reprojected) ** 2  # Squared error
        errors.append(error)
    
    error_array = np.array(errors).ravel() / num_points
    
    #print(np.sum(error_array))  
    return error_array

def bundle_adjustment(points_3D, uni_points2D, Rt_new, K, reprojection_tolerance):
    """
    Performs bundle adjustment to optimize camera parameters and 3D points.
    
    Args:
        points_3D : 3D points in the world coordinate system. 
                                Shape: (N, 3), where N is the number of points.
        uni_points2D : Corresponding 2D points in the image plane.
                                   Shape: (N, 2), where N is the number of points.
        Rt_new : Extrinsic parameters (Rotation and Translation) (3x4 matrix).
        K : Intrinsic camera matrix (3x3 matrix).
        reprojection_tolerance : Tolerance for reprojection error in optimization.

    Returns: 
    X
    image_points_2D
    Rt
    """
    # Combine all variables into array (not really sure what to name)
    optimization_variables = np.hstack((Rt_new.ravel(), K.ravel()))
    optimization_variables = np.hstack((optimization_variables, uni_points2D.ravel()))
    optimization_variables = np.hstack((optimization_variables, points_3D.ravel()))

    # Reprojection error
    error = np.sum(optimal_reprojection(optimization_variables))
    # print(f"Initial reprojection error: {error}")

    # Least-squares optimization
    optimization_result = least_squares(
        fun=optimal_reprojection, 
        x0=optimization_variables, 
        gtol=reprojection_tolerance
    )

    # Extract optimized parameters
    corrected_values = optimization_result.x
    corrected_extrinsics = corrected_values[0:12].reshape((3, 4))
    corrected_intrinsics = corrected_values[12:21].reshape((3, 3))
    
    remaining_points = len(corrected_values[21:])
    num_2D_points = int(remaining_points * 0.4)
    image_points_2D = corrected_values[21:21 + num_2D_points].reshape((2, num_2D_points // 2)).T
    image_points_3D = corrected_values[21 + num_2D_points:].reshape((-1, 3))
    image_points_2D = image_points_2D.T

    return points_3D, image_points_2D, Rt_new

    

def PnP(com_pts_3d, com_pts_left, com_pts_right, K, initial ,d = np.zeros((5, 1), dtype=np.float32)):
    """
    Recover camera rotation and translation from 3D and 2D points

    Args:
        X: 3D points
        p: Corresponding 2D points
        K: Camera intrinsic matrix
        d: distortion coefficient
        p_0: 

    """
    # if initial == 1:
    #     X = X[:, 0, :]
    #     p = p.T
    #     p_0 = p_0.T

    ret, rvecs, t, inliers = cv.solvePnPRansac(com_pts_3d, com_pts_right, K, d, cv.SOLVEPNP_ITERATIVE) # type: ignore

    # Convert from rotation vector to rotation matrix
    R, _ = cv.Rodrigues(rvecs)

    # Filter out bad 2D-3D correspondences
    if inliers is not None:
        com_pts_left = com_pts_left[inliers[:, 0]]
        com_pts_right= com_pts_right[inliers[:, 0]]
        com_pts_3d = com_pts_3d[inliers[:, 0]]
        
    return R, t, com_pts_3d, com_pts_left, com_pts_right


def common_points(right_pts_ref, left_pts, right_pts):

    com_idx_left = []
    com_idx_right = []
    for i in range(right_pts_ref.shape[0]):
        match = np.where(left_pts==right_pts_ref[i, :])
        if match[0].size == 0:
            pass
        else:
            com_idx_left.append(i)
            com_idx_right.append(match[0][0])
        
    unique_pts_left = np.ma.array(left_pts, mask=False)
    unique_pts_left.mask[com_idx_right] = True
    unique_pts_left = unique_pts_left.compressed()
    unique_pts_left = unique_pts_left.reshape(int(unique_pts_left.shape[0] / 2), 2)

    unique_pts_right = np.ma.array(right_pts, mask=False)
    unique_pts_right.mask[com_idx_right] = True
    unique_pts_right = unique_pts_right.compressed()
    unique_pts_right = unique_pts_right.reshape(int(unique_pts_right.shape[0]/2), 2)
        
    return np.array(com_idx_left), np.array(com_idx_right), unique_pts_left, unique_pts_right

if __name__ == "__main__":

    pass

