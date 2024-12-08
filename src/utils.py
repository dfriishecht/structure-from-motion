import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import os
from pathlib import Path

def capture_from_laptop():
    Path('capture/').mkdir(parents=True, exist_ok=True)
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access camera!")
        exit()
    i = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame. Exiting...")
            break
        
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('c'):
            cv.imwrite(f'capture/frame{i}.png', frame)
            i += 1
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows


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
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)
        
        if ret:
            objpoints.append(objp)  # Add 3D points
            imgpoints.append(corners)  # Add 2D points


    # Perform camera calibration to find the intrinsic matrix and distortion coefficients
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None) # type: ignore

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
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    capture_from_laptop()
