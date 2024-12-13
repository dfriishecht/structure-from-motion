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

downscale = 2.0
K[0,0] = K[0,0] / (downscale)
K[1,1] = K[1,1] / (downscale)
K[0,2] = K[0,2] / (downscale)
K[1,2] = K[1,2] / (downscale)

# Parameters
gtol_thresh = 0.5  # Gradient termination threshold for bundle adjustment
adjust_bundle = False  # Toggle for using bundle adjustment
img_dir = './gustav'  # Directory containing images
img_list = sorted([img for img in os.listdir(img_dir) if img.lower().endswith(('.jpg', '.png'))])
images = [cv.imread(os.path.join(img_dir, img)) for img in img_list]
images = [downsample(img, downscale) for img in images]
tot_imgs = len(images) - 2  # Total number of image pairs to process

# Initialize variables
pose0 = np.eye(3, 4)  # Identity matrix for the first frame's pose
pose1 = np.zeros((3, 4))  # Placeholder for the second frame's pose
point_cloud = np.zeros((1,3))  # 3D points will be stored here
P1 = np.matmul(K, pose0)

# Step 1: Process the first pair of images
img0 = images[0]
img1 = images[1]


# Find matched points between the first two images
_, left_pts_ref, right_pts_ref = match_keypoints_flann(img0, img1)

# Compute the Essential Matrix and filter inliers
E, mask = cv.findEssentialMat(left_pts_ref, right_pts_ref, K, method=cv.RANSAC, prob=0.999, threshold=0.4, mask = None)
left_pts_ref = left_pts_ref[mask.ravel() == 1]
right_pts_ref = right_pts_ref[mask.ravel() == 1]

# Recover pose (rotation and translation)
_, rot, t, mask = cv.recoverPose(E, left_pts_ref, right_pts_ref, K)

# Update the pose matrix
pose1[:3, :3] = np.matmul(rot, pose0[:3, :3])
pose1[:3, 3] = pose0[:3, 3] + np.matmul(pose0[:3, :3], t.ravel())

# Compute projection matrices
P_REF = P1
P2 = np.dot(K, pose1)

left_pts_ref = left_pts_ref[mask.ravel() > 0]
right_pts_ref = right_pts_ref[mask.ravel() > 0]

# Triangulate points and calculate reprojection error
points_3d = triangulate(P_REF, P2, left_pts_ref, right_pts_ref)
error, points_3d, _ = reprojection_error(points_3d, right_pts_ref, pose1, K)
rot, t, points_3d ,_ ,right_pts_ref = PnP(points_3d, left_pts_ref, right_pts_ref, K)
print(f"Initial reprojection error: {np.round(error,4)}")


# TODO start debugging from here
# Step 2: Process remaining images incrementally
prev_img = img1
for i, next_img in enumerate(tqdm(images[2:], desc="Processing images")):
    
    if i > 0:
        # Set new reference
        points_3d = triangulate(P1, P2, left_pts_ref, right_pts_ref)


    # Match keypoints between the previous and current image
    _, left_pts, right_pts = match_keypoints_flann(prev_img, next_img)


    # Find common points for PnP and triangulation
    com_idx_left, com_idx_right, unique_pts_left, unique_pts_right = common_points(right_pts_ref, left_pts, right_pts)
    com_pts_3d = points_3d[com_idx_left]
    com_pts_left = left_pts[com_idx_right]
    com_pts_right = right_pts[com_idx_right]

    # up to here seems to be working

    # Estimate new pose using PnP
    Rot, trans, com_pts_3d, com_pts_left, com_pts_right = PnP(com_pts_3d, com_pts_left, com_pts_right, K)
    pose_new = np.hstack((Rot, trans))
    P_new = np.matmul(K, pose_new)
    
    # Reprojection error for PnP result
    error, _, _ = reprojection_error(com_pts_3d, com_pts_right, pose_new, K)
    print(f"Reprojection error after PnP (image {i+3}): {error}")
    
    # Triangulate new points
    new_pts_3d = triangulate(P2, P_new, unique_pts_left, unique_pts_right)
    error, _, _ = reprojection_error(new_pts_3d, unique_pts_right, pose_new, K)
    print(f"Reprojection error for new points (image {i+3}): {error}")
    
    # Append new points to the point cloud
    point_cloud = np.vstack((point_cloud, new_pts_3d))


    # Update projection matrix and references
    prev_img = next_img
    P1 = P2
    P2 = P_new
    left_pts_ref = left_pts
    right_pts_ref = right_pts
    
output(point_cloud)
