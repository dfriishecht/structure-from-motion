"""
Incremental SFM
"""

import os
from tqdm import tqdm

from src.utils import *
from src.corres_search import *
from src.reconstruction import *

# Intrinsic for Gustav dataset
K = np.array([[2393.952166119461, -3.410605131648481e-13, 932.3821770809047], [0, 2398.118540286656, 628.2649953288065], [0, 0, 1]])

# Set parameters and initialize variables
densify = False  # Placeholder for densification; to be considered separately.
downscale = 2    # Example downscale factor; adjust as necessary.
gtol_thresh = 0.5  # Threshold for bundle adjustment gradient termination.
adjust_bundle = False  # Set to True if bundle adjustment is used.


# Load and sort images
img_dir = './gustav'
img_list = sorted([img for img in os.listdir(img_dir) if img.lower().endswith(('.jpg', '.png'))])
images = [cv.imread(os.path.join(img_dir, img)) for img in img_list]
tot_imgs = len(images) - 2  # Number of image pairs to process


# ---------------------- Main loop starts here ------------------------------------ #

# Initialize camera matrices and transformations
pose0 = np.eye(3, 4)  # Initial pose for the first frame
pose1 = np.zeros((3, 4))  # Placeholder for subsequent poses
point_cloud = None  # 3D points
colorstot = np.zeros((1, 3))  # Corresponding colors


img0 = images[0]
img1 = images[1]

# # downscale if needed
# img0 = img_downscale(images[0], downscale)
# img1 = img_downscale(images[1], downscale)

left_pts_ref, right_pts_ref = match_keypoints_flann(img0, img1)

# Compute the essential matrix and recover pose
E, mask = cv.findEssentialMat(left_pts_ref, right_pts_ref, K, method=cv.RANSAC, prob=0.999, threshold=0.4) # type: ignore

left_pts_ref = left_pts_ref[mask.ravel() == 1] # type: ignore
right_pts_ref = right_pts_ref[mask.ravel() == 1] # type: ignore

# Recover rotation & translation using Essential matrix
_, R, t, mask = cv.recoverPose(E, left_pts_ref, right_pts_ref, K)
left_pts_ref = left_pts_ref[mask.ravel() > 0]
right_pts_ref = right_pts_ref[mask.ravel() > 0]

# Update pose and find projection matrices
pose1[:3, :3] = np.dot(R, pose0[:3, :3])
pose1[:3, 3] = pose0[:3, 3] + np.dot(pose0[:3, :3], t.ravel())
P1 = np.dot(K, pose0)
P2 = np.dot(K, pose1)

# Get 3d points for first image pair
points_3d = triangulate(P1, P2, left_pts_ref, right_pts_ref)
error, points_3d, repro_pts = reprojection_error(points_3d, right_pts_ref, pose1, K, homogenity = 0)

print(points_3d.dtype, right_pts_ref.dtype)
print("REPROJECTION ERROR: ", error)

point_cloud = points_3d

# plot_3d(point_cloud)

prev_img = img1
# Expand the 3D point cloud by processing the remaining image incrementally
for next_img in tqdm(images[2:]):
    
    left_pts, right_pts = match_keypoints_flann(prev_img, next_img)

    # Find common points to use for PnP, also filter out unique pts only to triangulate
    com_idx_left, com_idx_right, unique_pts_left, unique_pts_right = common_points(right_pts_ref, left_pts, right_pts)

    com_pts_ref = right_pts_ref[com_idx_left]
    com_pts_left = left_pts[com_idx_right]
    com_pts_right = right_pts[com_idx_right]
    com_pts_3d = points_3d[com_idx_left]
    

    # Estimate new image's pose using PnP
    Rot, trans, points_3d, com_pts_left, com_pts_right = PnP(com_pts_3d, com_pts_left, com_pts_right, K, initial=0)
    pose_new = np.hstack((Rot, trans))
    Pnew = np.dot(K, pose_new)


    # Reprojection error
    error, _ , _ = reprojection_error(points_3d, com_pts_left, pose_new, K, homogenity=0)
    print(error)
    

    new_pts_3d = triangulate(P2, Pnew, unique_pts_left, unique_pts_right)
    error, _ , _ = reprojection_error(new_pts_3d, unique_pts_right, pose_new, K, homogenity=0)
    print(error)

    print(point_cloud.shape)
    point_cloud = np.vstack((point_cloud, new_pts_3d))
    print(point_cloud.shape)
    plot_3d(point_cloud)

    # Bundle Adjustment
    points_3d, temp2, Rtnew = bundle_adjustment(points_3d, temp2, Rtnew, K, gtol_thresh)
    Pnew = np.dot(K, Rtnew)
    error, points_3d, _ = reprojection_error(points_3d, temp2, Rtnew, K, homogenity=0)


    # # Update poses and image data
    # pose0 = pose1.copy()
    # P1, P2 = P2, Pnew
    
    # prev_img = next_img
    # left_pts_ref, right_pts_ref = pts_, left_pts


#     # Display current image
#     cv.imshow('image', next_img)
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

# cv.destroyAllWindows()




