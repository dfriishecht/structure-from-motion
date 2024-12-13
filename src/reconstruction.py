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
    Triangulate two sets of 2D points into one set of 3D coordinates using the Direct Linear Transformation (DLT) method.

    Args:
        P0 (np.ndarray): 3x4 projection matrix for the first image.
        P1 (np.ndarray): 3x4 projection matrix for the second image.
        pts0 (np.ndarray): Nx2 array of 2D points from the first image.
        pts1 (np.ndarray): Nx2 array of 2D points from the second image.

    Returns:
        np.ndarray: Nx3 array of 3D points in Cartesian coordinates.
    """

    def get_A_mat(uv0: np.ndarray, uv1: np.ndarray) -> np.ndarray:
        """
        Construct the matrix A for the linear system AX = 0 to triangulate a point.

        Args:
            uv0 (np.ndarray): A 1x2 array [u, v] of a 2D point from the first image.
            uv1 (np.ndarray): A 1x2 array [u, v] of a 2D point from the second image.

        Returns:
            np.ndarray: A 4x4 matrix A for solving the triangulation problem.
        """
        u0, v0 = uv0[0], uv0[1]
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
        _, _, V = np.linalg.svd(A_mat)  # Perform Singular Value Decomposition
        X_hom = V[-1]  # Homogeneous solution (last row of V)
        X_cartesian = X_hom[0:3] / X_hom[3]  # Convert from homogeneous to Cartesian coordinates
        points_3d.append(X_cartesian)

    return np.array(points_3d)


def reprojection_error(X: np.ndarray, pts: np.ndarray, Rt: np.ndarray, K: np.ndarray):
    """
    Calculates the error between 2D image points and the corresponding 3D
    points projected back down to the image plane.

    Args:
        X (np.ndarray): 4xN matrix of 3D points in homogeneous coordinates.
        pts (np.ndarray): Nx2 array of 2D image points that correspond to 3D points in X.
        Rt (np.ndarray): A 3x4 matrix representing the camera pose (rotation R and translation t).
        K (np.ndarray): The 3x3 camera intrinsic matrix.

    Returns:
        float: The average reprojection error.
        np.ndarray: The original 3D points in homogeneous coordinates.
        np.ndarray: The 2D projected points.
    """
    R = Rt[:3, :3]  # Extract rotation matrix
    translation = Rt[:3, 3]  # Extract translation vector

    # Convert rotation matrix to Rodrigues vector for OpenCV compatibility
    rotation, _ = cv.Rodrigues(R)

    # Project 3D points to 2D
    proj, _ = cv.projectPoints(X, rotation, translation, K, distCoeffs=None) # type: ignore
    proj = np.float32(proj[:, 0, :])  # Reshape to Nx2 array

    # Compute the L2 norm of the difference between original and projected points
    error = cv.norm(proj, pts, cv.NORM_L2)  # type: ignore # Total reprojection error
    error /= len(proj)  # type: ignore # Normalize by the number of points

    return error, X, proj


def optimal_reprojection(x):
    """
    Calculates the reprojection error for a given set of parameters.

    Args: 
        x (np.ndarray): 
            A 1D array containing the following concatenated elements:
            - First 12 values: Extrinsic matrix (3x4) flattened row-wise.
            - Next 9 values: Intrinsic matrix (3x3) flattened row-wise.
            - Next values: 2D points followed by 3D points.
              The number of 2D points is 40% of the remaining elements,
              and the number of 3D points is 60%. The 2D points are 
              represented as (u, v), and the 3D points are represented as (x, y, z).

    Returns:
        np.ndarray: 
            A 1D array of normalized reprojection errors for each point pair.
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
    points_2D, _ = cv.projectPoints(X, rotation_vector, t, K, distCoeffs=None)  # type: ignore 
    points_2D = points_2D[:, 0, :]  #  (N, 2) for 2D

    errors = []
    image_points = image_points.T  # Convert to shape (N, 2)
    num_points = len(image_points)

    for index in range(num_points):
        observed = image_points[index]  # Existing points
        reprojected = points_2D[index]  # New points
        error = (observed - reprojected) ** 2  # Squared error
        errors.append(error)
    
    error_array = np.array(errors).ravel() / num_points
    
    return error_array


def bundle_adjustment(points_3D, uni_points2D, Rt_new, K, reprojection_tolerance):
    """
    Performs bundle adjustment to optimize camera parameters and 3D points.

    Args:
        points_3D (np.ndarray): 
            A 2D array of 3D points in the world coordinate system.
            Shape: (N, 3), where N is the number of points.
        uni_points2D (np.ndarray): 
            A 2D array of corresponding 2D points in the image plane.
            Shape: (N, 2), where N is the number of points.
        Rt_new (np.ndarray): 
            A 3x4 matrix representing the extrinsic parameters, 
            which include rotation and translation.
        K (np.ndarray): 
            A 3x3 intrinsic camera matrix.
        reprojection_tolerance (float): 
            Tolerance for the reprojection error used in the optimization.

    Returns:
        tuple: A tuple containing the following:
            - np.ndarray: Optimized 3D points in the world coordinate system 
                          (Shape: (N, 3)).
            - np.ndarray: Optimized 2D image points in the image plane 
                          (Shape: (N, 2)).
            - np.ndarray: Updated extrinsic parameters (3x4 matrix).
    """
    # Combine all variables into a single array
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


def PnP(com_pts_3d, com_pts_left, com_pts_right, K, d=np.zeros((5, 1), dtype=np.float32)):
    """
    Estimates the camera pose (rotation and translation) using the Perspective-n-Point (PnP) algorithm.
    
    Args:
        com_pts_3d (np.ndarray): 
            3D points in the world coordinate system. Shape: (N, 3), where N is the number of points.
        com_pts_left (np.ndarray): 
            2D points in the left image plane. Shape: (N, 2), where N is the number of points.
        com_pts_right (np.ndarray): 
            2D points in the right image plane. Shape: (N, 2), where N is the number of points.
        K (np.ndarray): 
            Camera intrinsic matrix. Shape: (3, 3).
        d (np.ndarray, optional): 
            Distortion coefficients for the camera. Defaults to a zero array of shape (5, 1).

    Returns:
        tuple: A tuple containing the following:
            - R (np.ndarray): Rotation matrix (3x3) representing the camera's orientation.
            - t (np.ndarray): Translation vector (3x1) representing the camera's position.
            - com_pts_3d (np.ndarray): Filtered 3D points after removing outliers. Shape: (M, 3), where M ≤ N.
            - com_pts_left (np.ndarray): Filtered 2D points in the left image plane. Shape: (M, 2).
            - com_pts_right (np.ndarray): Filtered 2D points in the right image plane. Shape: (M, 2).
    """
    _, rvecs, t, inliers = cv.solvePnPRansac(com_pts_3d, com_pts_right, K, d, cv.SOLVEPNP_ITERATIVE)  # type: ignore

    # Convert from rotation vector to rotation matrix
    R, _ = cv.Rodrigues(rvecs)

    # Filter out bad 2D-3D correspondences
    if inliers is not None:
        com_pts_left = com_pts_left[inliers[:, 0]]
        com_pts_right = com_pts_right[inliers[:, 0]]
        com_pts_3d = com_pts_3d[inliers[:, 0]]
        
    return R, t, com_pts_3d, com_pts_left, com_pts_right


def common_points(right_pts_ref, left_pts, right_pts):
    """
    Identifies common 2D points between two sets of image points and extracts unique points.

    Args:
        right_pts_ref (np.ndarray): 
            Reference 2D points in the right image. Shape: (N, 2), where N is the number of points.
        left_pts (np.ndarray): 
            2D points in the left image. Shape: (M, 2), where M is the number of points.
        right_pts (np.ndarray): 
            2D points in the right image. Shape: (M, 2), where M is the number of points.

    Returns:
        tuple: A tuple containing:
            - com_idx_left (np.ndarray): 
                Indices of common points in `right_pts_ref` that match with `left_pts`.
            - com_idx_right (np.ndarray): 
                Indices of corresponding points in `left_pts` that match `right_pts_ref`.
            - unique_pts_left (np.ndarray): 
                Unique 2D points in the left image not matching `right_pts_ref`.
                Shape: (K, 2), where K is the number of unique points.
            - unique_pts_right (np.ndarray): 
                Unique 2D points in the right image not matching `right_pts_ref`.
                Shape: (L, 2), where L is the number of unique points.
    """
    com_idx_left = []
    com_idx_right = []
    
    for i in range(right_pts_ref.shape[0]):
        match = np.where(left_pts == right_pts_ref[i, :])
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
    unique_pts_right = unique_pts_right.reshape(int(unique_pts_right.shape[0] / 2), 2)
        
    return np.array(com_idx_left), np.array(com_idx_right), unique_pts_left, unique_pts_right


if __name__ == "__main__":

    pass

