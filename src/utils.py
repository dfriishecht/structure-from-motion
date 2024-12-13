import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import os
from pathlib import Path

def downsample(img, downscale):
	downscale = int(downscale/2)
	i = 0
	while(i < downscale):
		img = cv.pyrDown(img)
		i += 1
	return img


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


def save_to_ply(point_cloud, filename="output.ply", colors=None):
    """
    Save the 3D point cloud to a .ply file.
    
    Parameters:
        point_cloud (numpy.ndarray): Array of shape (N, 3) containing 3D points.
        filename (str): Name of the output .ply file.
        colors (numpy.ndarray, optional): Array of shape (N, 3) containing RGB colors for each point.
    """
    assert point_cloud.shape[1] == 3, "Point cloud must have shape (N, 3)"
    
    if colors is None:
        # Default to white if no colors are provided
        colors = np.full((point_cloud.shape[0], 3), 255, dtype=np.uint8)

    assert colors.shape[0] == point_cloud.shape[0], "Colors must have the same number of points"
    assert colors.shape[1] == 3, "Colors must have shape (N, 3)"

    # Header for the .ply file
    ply_header = f"""ply
                    format ascii 1.0
                    element vertex {len(point_cloud)}
                    property float x
                    property float y
                    property float z
                    property uchar red
                    property uchar green
                    property uchar blue
                    end_header
                    """
    with open(filename, 'w') as ply_file:
        ply_file.write(ply_header)
        for point, color in zip(point_cloud, colors):
            ply_file.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")

    print(f"Point cloud saved to {filename}")


def output(final_cloud,pixel_colour = None):
    if pixel_colour:
        output_colors=pixel_colour.reshape(-1, 3)
    else:
        output_colors = np.full((final_cloud.shape[0], 3), 255, dtype=np.uint8)
        
    output_points=final_cloud.reshape(-1, 3) * 200
    
    mesh=np.hstack([output_points,output_colors])

    mesh_mean=np.mean(mesh[:,:3],axis=0)
    diff=mesh[:,:3]-mesh_mean
    distance=np.sqrt(diff[:,0]**2+diff[:,1]**2+diff[:,2]**2)
    
    index=np.where(distance<np.mean(distance)+300)
    mesh=mesh[index]
    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar blue
        property uchar green
        property uchar red
        end_header
        '''
    with open('sparse.ply', 'w') as f:
        f.write(ply_header % dict(vert_num=len(mesh)))
        np.savetxt(f,mesh,'%f %f %f %d %d %d')
    print("Point cloud processed, cleaned and saved successfully!")



if __name__ == "__main__":
    capture_from_laptop()
