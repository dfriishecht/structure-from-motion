import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob


def get_intrinsic(device_name: str) -> tuple:
    images = glob.glob(f"chess/{device_name}/*")

    # Define chessboard dimensions
    chessboard_size = (9, 6)  # Number of inner corners per chessboard row and column
    square_size = 29/32  # The actual size of a square on your chessboard (e.g., in cm or m)

    # Prepare object points (3D points in real world space)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # Arrays to store object points and image points from all images
    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane

    # Loop over your images
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        if ret:
            objpoints.append(objp)  # Add 3D points
            imgpoints.append(corners)  # Add 2D points


    # Perform camera calibration to find the intrinsic matrix and distortion coefficients
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None) # type: ignore

    # Print out the intrinsic matrix
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs)
    return camera_matrix, dist_coeffs




def plot_3d(pts_3d: np.ndarray):
    # Split into x, y, z components
    x = pts_3d[:, 0]
    y = pts_3d[:, 1]
    z = pts_3d[:, 2]

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=z, cmap='viridis')  # Color points by z-value

    # Add labels and show the plot
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


if __name__ == "__main__":
    data = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
    ])
    plot_3d(data)
