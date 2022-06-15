import cv2
import numpy as np
import os
import glob
import sys

# Defining the dimensions of checkerboard.
# It must be equal to the number of corners that the Canny86 algorithm find in each images.
w = 9
h = 6
CHECKERBOARD = (h, w)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D world points for each checkerboard image.
objpoints = []
# Creating vector to store vectors of 2D image points for each checkerboard image.
imgpoints = []

# Defining the world coordinates for 3D points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
#print(np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].shape)
#print(np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.shape)
#print(np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2).shape)
#prev_img_shape = None
#print(objp.shape)
#sys.exit()

# Extracting path of individual image stored in a given directory
raw_dir = './images_raw/'
before_dir = './images_before_rectifying/'
after_dir = './images_after_rectifying/'
undist_dir = './images_undistorted/'
fnames_raw = os.listdir(raw_dir)
fnames_undist = os.listdir(undist_dir)

for fname in fnames_raw:
    #print(raw_dir+fname)

    raw_img = cv2.imread(raw_dir+fname)
    gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    # The reference for findChessboardCorners method is not mentioned in the official doc.
    # Instead, the reference for findChessboardCornersSB is mentioned, which is more accurate and faster than findChessboardCorners method. 
    # Alexander Duda and Udo Frese. Accurate detection and localization of checkerboard corners for calibration. In 29th British Machine Vision Conference. British Machine Vision Conference (BMVC-29), September 3-6, Newcastle, United Kingdom. BMVA Press, 2018.
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, \
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK\
        + cv2.CALIB_CB_NORMALIZE_IMAGE)

    """
    If desired numbers of the corners are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    img_before_cal = cv2.drawChessboardCorners(raw_img, CHECKERBOARD, corners, ret)
    #cv2.imshow('img',img)
    #cv2.waitKey(0)

    cv2.imwrite(f'{before_dir}{fname}', img_before_cal)

    #sys.exit()

    if ret is True:
        objpoints.append(objp)

        # refining pixel coordinates for given 2d points.
        # According to the official opencv doc, cornerSubPix method is based on
        # W FORSTNER. A fast operator for detection and precise location of distincs points, corners and center of circular features. In Proc. of the Intercommission Conference on Fast Processing of Photogrammetric Data, Interlaken, Switzerland, 1987, pages 281–305, 1987.
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        #print(corners.shape)
        #print(corners2.shape)
        imgpoints.append(corners2)

        # Draw and display the corners
        img_after_cal = cv2.drawChessboardCorners(raw_img, CHECKERBOARD, corners2, ret)

        cv2.imwrite(f'{after_dir}{fname}', img_after_cal)

    else: 
        #raise ValueError(f"The corner of {fname} is not found.")
        print(f"The corner of {fname} is not found.")
        continue

    #else: raise ValueError("Check the dimension of the chessboard or change the chessboard image which one can identify the corners more clearly.")

    #cv2.imshow('img',img)
    #cv2.waitKey(0)

cv2.destroyAllWindows()

h, w = raw_img.shape[:2]
print(h,w)
# Performing camera calibration by 
# passing the value of known 3D points (objpoints)
# and corresponding pixel coordinates of the 
# detected corners (imgpoints)

#print(gray.shape, gray.shape[::-1])

# According to the official opence doc, calibrateCamera method is based on
# 
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)

# Using the derived camera parameters to undistort the image

for fname in fnames_raw:

    img = cv2.imread(raw_dir+fname)
    # Refining the camera matrix using parameters obtained by calibration
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Method 1 to undistort the image
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # Method 2 to undistort the image
    #mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    #dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # Displaying the undistorted image
    #cv2.imshow("undistorted image", dst)
    #cv2.waitKey(0)

    # Save the undistorted image.
    cv2.imwrite(f'{undist_dir}{fname}', dst)