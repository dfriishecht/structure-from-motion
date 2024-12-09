import os
import numpy as np
import cv2 as cv
from tqdm import tqdm
from src.utils import *
from src.corres_search import *
from src.reconstruction import *

# Intrinsic matrix for the Gustav dataset
K = np.array([
    [2393.952166119461, -3.410605131648481e-13, 932.3821770809047],
    [0, 2398.118540286656, 628.2649953288065],
    [0, 0, 1]
])

# Parameters
gtol_thresh = 0.5  # Gradient termination threshold for bundle adjustment
adjust_bundle = False  # Toggle for using bundle adjustment
img_dir = './gustav'  # Directory containing images
img_list = sorted([img for img in os.listdir(img_dir) if img.lower().endswith(('.jpg', '.png'))])
images = [cv.imread(os.path.join(img_dir, img)) for img in img_list]
tot_imgs = len(images) - 2  # Total number of image pairs to process

# Initialize variables
pose0 = np.eye(3, 4)  # Identity matrix for the first frame's pose
pose1 = np.zeros((3, 4))  # Placeholder for the second frame's pose
point_cloud = None  # 3D points will be stored here

# Step 1: Process the first pair of images
img0 = images[0]
img1 = images[1]

# Match keypoints between the first two images
left_pts_ref, right_pts_ref = match_keypoints_flann(img0, img1)

# Compute the Essential Matrix and filter inliers
E, mask = cv.findEssentialMat(left_pts_ref, right_pts_ref, K, method=cv.RANSAC, prob=0.999, threshold=0.4)
left_pts_ref = left_pts_ref[mask.ravel() == 1]
right_pts_ref = right_pts_ref[mask.ravel() == 1]

# Recover pose (rotation and translation)
_, R, t, mask = cv.recoverPose(E, left_pts_ref, right_pts_ref, K)
left_pts_ref = left_pts_ref[mask.ravel() > 0]
right_pts_ref = right_pts_ref[mask.ravel() > 0]

# Update the pose matrix
pose1[:3, :3] = np.dot(R, pose0[:3, :3])
pose1[:3, 3] = pose0[:3, 3] + np.dot(pose0[:3, :3], t.ravel())

# Compute projection matrices
P1 = np.dot(K, pose0)
P2 = np.dot(K, pose1)

# Triangulate points and calculate reprojection error
points_3d = triangulate(P1, P2, left_pts_ref, right_pts_ref)
error, points_3d, _ = reprojection_error(points_3d, right_pts_ref, pose1, K, homogenity=0)
print(f"Initial reprojection error: {error}")

point_cloud = points_3d  # Initialize point cloud

# Step 2: Process remaining images incrementally
prev_img = img1
for i, next_img in enumerate(tqdm(images[2:], desc="Processing images")):
    # Match keypoints between the previous and current image
    left_pts, right_pts = match_keypoints_flann(prev_img, next_img)

    if i != 0:
        points_3d = triangulate(P1, P2, left_pts_ref, right_pts_ref)

    # Find common points for PnP and triangulation
    com_idx_left, com_idx_right, unique_pts_left, unique_pts_right = common_points(right_pts_ref, left_pts, right_pts)
    print(points_3d.shape)
    print(com_idx_left)
    com_pts_3d = points_3d[com_idx_left]
    com_pts_ref = right_pts_ref[com_idx_left]
    com_pts_left = left_pts[com_idx_right]
    com_pts_right = right_pts[com_idx_right]

    # Estimate new pose using PnP
    Rot, trans, com_pts_3d, com_pts_left, com_pts_right = PnP(com_pts_3d, com_pts_left, com_pts_right, K, initial=0)
    pose_new = np.hstack((Rot, trans))
    P_new = np.dot(K, pose_new)

    # Reprojection error for PnP result
    error, _, _ = reprojection_error(com_pts_3d, com_pts_right, pose_new, K, homogenity=0)
    print(f"Reprojection error after PnP (image {i+2}): {error}")

    # Triangulate new points
    new_pts_3d = triangulate(P2, P_new, unique_pts_left, unique_pts_right)
    error, _, _ = reprojection_error(new_pts_3d, unique_pts_right, pose_new, K, homogenity=0)
    print(f"Reprojection error for new points (image {i+2}): {error}")

    # Append new points to the point cloud
    point_cloud = np.vstack((point_cloud, new_pts_3d))

    # Optionally perform bundle adjustment
    if adjust_bundle:
        new_pts_3d, right_pts_adjusted, Rt_new = bundle_adjustment(new_pts_3d, com_pts_right, pose_new, K, gtol_thresh)
        P_new = np.dot(K, Rt_new)
        error, _, _ = reprojection_error(new_pts_3d, right_pts_adjusted, Rt_new, K, homogenity=0)
        print(f"Reprojection error after bundle adjustment (image {i+2}): {error}")

    # Update projection matrix and references
    P2 = np.copy(P_new)
    left_pts_ref = np.copy(left_pts)
    right_pts_ref = np.copy(right_pts)
    prev_img = np.copy(next_img)
    break


save_to_ply(point_cloud, filename='output_1.ply')
