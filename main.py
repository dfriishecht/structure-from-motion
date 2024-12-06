"""
Incremental SFM
"""

import os
import open3d as o3d

from src.corres_search import *
from src.reconstruction import *

K = np.array([[2393.952166119461, -3.410605131648481e-13, 932.3821770809047], [0, 2398.118540286656, 628.2649953288065], [0, 0, 1]])

img_dir = 'gustav/'
img_list = sorted(os.listdir(img_dir))
images = []
for img in img_list:
    if '.jpg' in img.lower() or '.png' in img.lower():
        images = images + [img]
i = 0

if __name__ == "__main__":

    posearr = K.ravel()

    # Initialize the first 2 images as reference images
    pose_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]) # Assuming first image is at world coords origin
    pose_1 = np.empty((3, 4))

    P1 = np.matmul(K, pose_0)
    Pref = P1
    P2 = np.empty((3, 4))

    Xtot = np.zeros((1, 3))
    colorstot = np.zeros((1, 3))

    print(img_list)





