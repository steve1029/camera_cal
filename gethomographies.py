import os, sys
from pickletools import optimize
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy.optimize as spo

"""The homography matrix is calculated.
The 3D world points are given with Z=0.
The coordinates of each image points are 
given by cv2.findChessboardCornersSB() method.
"""

def get_world_points(CHECKERBOARD, raw_dir, save_dir):

    # Creating vector to store vectors of the 3D world points for each checkerboard image.
    # Unit: centimeter.
    # (w, h, k) order. Note that all k=0.
    wps = []
    wp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    wp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) 

    #print(wp)

    # Creating vector to store vectors of the 2D sensor points for each checkerboard image.
    # Note that this vector lies in the sensor coordinates, not the image plane.
    ips = [] # findChessboardCornersSB()

    # Extracting path of individual image stored in a given directory
    if os.path.exists(save_dir) is False: os.makedirs(dir)

    fnames_raw = os.listdir(raw_dir)
    fnames_found = []

    for fname in fnames_raw:
        #print(raw_dir+fname)

        raw_img = cv2.imread(raw_dir+fname)
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD, \
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK\
            + cv2.CALIB_CB_NORMALIZE_IMAGE)

        """
        If desired numbers of the corners are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """

        if ret is True:

            #raw_img = cv2.imread(raw_dir+fname)
            #img = cv2.drawChessboardCorners(raw_img, CHECKERBOARD, corners, ret)
            #cv2.imwrite(f'{strategy3_dir}{fname}', img)

            fnames_found.append(fname)
            wps.append(wp)
            #ips.append(corners)
            ips.append(np.squeeze(corners))

        else: 
            #raise ValueError(f"The corner of {fname} is not found.")
            print(f"The corner of {fname} is not found.")
            continue

    (h, w) = gray.shape[::-1]

    return wps, ips, fnames_found, (h,w)

def get_homographies(wps, ips):

    M = len(wps) # The number of the views, i.e. the images.
    Hs = []

    for m, (wp, ip) in enumerate(zip(wps,ips)):

        H_init = estimate_homography(wp, ip)
        opt_H, jac = refine_homography(H_init, wp, ip)
        Hs.append(opt_H)

    return Hs

def estimate_homography(wp, ip):
    """Estimate the homography matrix for each image

    Parameters
    ----------
    wps: a list.
        a list of 2D arrays. a world points of the chessboard corners.
        Note that in Zhang's technique, Z is omitted for the world coordinates
        because Z=0 for all chessboard corners.

    ips: a list.
        a list of 2D vectors, an observed sensor points in an images.

    Returns
    -------
    H: ndarray.
        3x3 ndarray. A homography matrix of an image.
    """

    N = wp.shape[0] # The number of points in an image.
    Np = get_normalization_matrix(wp)
    Nq = get_normalization_matrix(ip)
    M = np.zeros((2*N,9), dtype=np.float32)

    homwp = wp.copy()
    homip = ip.copy()

    homwp[:,2] = 1
    homip = np.append(homip, np.ones((54,1)), axis=1)

    for j, (p, q) in enumerate(zip(homwp, homip)):

        # Note that p[0] and q[0] are the horizental location and
        # p[1] and q[1] are the vertical location of the pixel. 

        pp = np.dot(Np, p)
        pp /= pp[2]
        pp = np.delete(pp, -1)

        qq = np.dot(Nq, q)
        qq /= qq[2]
        qq = np.delete(qq, -1)

        M[2*j  ,0] = -pp[0]
        M[2*j  ,1] = -pp[1]
        M[2*j  ,2] = -1
        M[2*j  ,6] = qq[0]*pp[0]
        M[2*j  ,7] = qq[0]*pp[1]
        M[2*j  ,8] = qq[0]

        M[2*j+1,3] = -pp[0]
        M[2*j+1,4] = -pp[1]
        M[2*j+1,5] = -1
        M[2*j+1,6] = qq[1]*pp[0]
        M[2*j+1,7] = qq[1]*pp[1]
        M[2*j+1,8] = qq[1]

    # Each M corresponds to one image.
    U, S, Vh = np.linalg.svd(M)
    h = Vh[np.argmin(S)]

    HH = h.reshape(3,3)
    H = np.linalg.multi_dot([np.linalg.inv(Nq), HH, Np])

    return H

def get_normalization_matrix(Xs):
    """
    Parameters
    ----------
    Xs: ndarray
        An array of 2D vector coordinates. Xs[j] = np.array([x,y]).

    Returns
    -------
    NM: 3x3 ndarray.
        A normalisation matrix.
    """
    N = Xs.shape[0]
    NM = np.zeros((3,3))

    xbar = 0
    ybar = 0

    xvar = 0
    yvar = 0

    # Calculate the centroid and variance.
    # There are two ways that give the same result.
    # Method 1: using for-loop.
    """
    for n in range(N):

        xbar += 1/N * Xs[n,0]
        ybar += 1/N * Xs[n,1]

    for n in range(N):

        xvar += 1/N * (Xs[n,0]-xbar)**2
        yvar += 1/N * (Xs[n,1]-ybar)**2

    print(xbar, ybar)
    print(xvar, yvar)
    """

    # Method 2: using the generator.
    xbar = 1/N * np.sum(np.fromiter((X[0] for X in Xs), np.float64))
    ybar = 1/N * np.sum(np.fromiter((X[1] for X in Xs), np.float64))

    xvar = 1/N * np.sum(np.fromiter(((X[0]-xbar)**2 for X in Xs), np.float64))
    yvar = 1/N * np.sum(np.fromiter(((X[1]-ybar)**2 for X in Xs), np.float64))

    #print(xbar, ybar)
    #print(xvar, yvar)

    sx = np.sqrt(2/xvar)
    sy = np.sqrt(2/yvar)

    NM[0,0] = sx
    NM[0,2] = -sx * xbar
    NM[1,1] = sy
    NM[1,1] = -sy * ybar
    NM[2,2] = 1

    return NM

def refine_homography(H, wp, ip):
    """
    Returns
    -------
    optH: ndarray
        3x3 homography matrix. Numerically optimized.
    """

    N = wp.shape[0] # The number of corners.

    M = np.zeros(2*N, dtype=np.float64) # observed corner locations, in sensor coordinate.

    for j, q in enumerate(ip):

        M[2*j  ] = q[0]
        M[2*j+1] = q[1]

    h = H.reshape(-1)

    result = spo.least_squares(residuals, h, method='lm', args=(M, wp))
    #result = spo.leastsq(residuals, h, args=(M, wp)) # MINPACK wrapper.

    # return the optimized homography.
    hh = result.x
    #hh = result[0]
    hh /= hh[8]
    hh = hh.reshape(3,3)

    J = result.jac

    return hh, J

def residuals(h, M, wp):

    Y = value(wp, h) # the estimated corner locations for an image, in sensor coordinate.
    E = M - Y # a function to minimize.
    
    return E

def value(wp, h):
    """Estimate the coordinate of the image points that lies in the sensor coordinate.

    Parameters
    ----------
    wp: ndarray.
        An 2D array containing an 3D vector coordinate in each row. 
        ex) wp[j] = np.array([x,y,z]).

    h: ndarray.
        A flattened homography matrix.

    Returns
    -------
    Y: ndarray
        a 1D vector. Y = (u0, v0, u1, v1, ...) where (u_i, v_i) is the
        coordinate of an image point in the sensor coordinate.
        Note that the sensor coordinate is one of the homogeneous coordinate,
        where Z=0.
    """

    N = wp.shape[0] # The number of corners.
    Y = np.zeros(2*N, np.float64)

    for j in range(N):

        # (x,y,z) is the world coordinate.
        (x,y,z) = wp[j]

        w = h[6]*x + h[7]*y + h[8]
        u = (h[0]*x + h[1]*y + h[2]) / w
        v = (h[3]*x + h[4]*y + h[5]) / w

        Y[2*j  ] = u
        Y[2*j+1] = v

    return Y

def jac(h, wp):
    """Get Jacobian matrix of size 2N x 9 where N is the number of corners.
    
    Parameters
    ----------
    wp: ndarray.
        An 2D array containing an 3D vector coordinate in each row. 
        ex) wp[j] = np.array([x,y,z]).

    h: ndarray.
        A flattened homography matrix.
    """

    N = wp.shape[0] # The number of corners.
    J = np.zeros((2*N,9), np.float64)

    for j, p in enumerate(wp):
        (X,Y,Z) = p
        sx = h[0]*X + h[1]*Y + h[2]
        sy = h[3]*X + h[4]*Y + h[5]
        w  = h[6]*X + h[7]*Y + h[8]

        J[2*j  ] = [X/w, Y/w, 1/w, 0, 0, 0, -sx*X/w**2, -sx*Y/w**2, -sx/w**2]
        J[2*j+1] = [0, 0, 0, X/w, Y/w, 1/w, -sy*X/w**2, -sy*Y/w**2, -sy/w**2]

    return J

def homography_compare(wp, ip):

    # The homography obtained by using built-in function in OpenCV.
    cv2_H, status = cv2.findHomography(wp, ip)

    # The homography optained by me.
    H_init = estimate_homography(wp, ip)
    opt_H, J = refine_homography(H_init, wp, ip)

    N = wp.shape[0] # The number of corners.

    Errs = np.zeros((N,2), dtype=np.float64)

    for n, (src, dst) in enumerate(zip(wp,ip)):

        hom_src = src.copy() # a second point in the first image.
        hom_src[2] = 1 # conversion to homogeneous coordinates.

        cv2_ip = np.dot(cv2_H, hom_src) # an estimated image point of wp01.
        opt_ip = np.dot(opt_H, hom_src) # an estimated image point of wp01.

        cv2_E = np.linalg.norm(ip[n]-(cv2_ip/cv2_ip[2])[:-1])
        opt_E = np.linalg.norm(ip[n]-(opt_ip/opt_ip[2])[:-1])

        Errs[n] = (cv2_E, opt_E)

    return Errs

if __name__ == '__main__':

    # Defining the dimensions of checkerboard.
    w = 9
    h = 6
    CHECKERBOARD = (h, w)
    raw_dir = './images_raw/'
    save_dir = './strategy3/'

    wps, ips, fnames, (h_npixels, w_npixels) = get_world_points(CHECKERBOARD, raw_dir, save_dir)

    #print(fnames_found[0])
    #print(ips[0].shape)
    #print(np.squeeze(ips[0]))
    #h, w = raw_img.shape[:2]
    #print(h,w)

    #H = estimate_homography(wps[0], ips[0])
    #opt_H, numJ = refine_homography(H, wps[0], ips[0])

    #print(H/H[2,2])
    #print(opt_H)

    #J = jac(H.reshape(-1), wps[0])

    Hs = get_homographies(wps, ips)

    Errs0 = homography_compare(wps[0], ips[0])
    #cv2_E1, opt_E1 = homography_compare(wps[1], ips[1])

    #print(cv2_E0, opt_E0)
    #print(cv2_E1, opt_E1)

    """
    opt_H0 = Hs[0]
    opt_H1 = Hs[1]
    opt_H2 = Hs[2]

    ret, intr, dist, rvecs, tvecs = cv2.calibrateCamera(wps, ips, (h_npixels, w_npixels), None, None)
    cv2_H0, status = cv2.findHomography(wps[0], ips[0])
    cv2_H1, status = cv2.findHomography(wps[1], ips[1])
    cv2_H2, status = cv2.findHomography(wps[2], ips[2])

    r = R.from_mrp(np.squeeze(rvecs[0]))
    rot = r.as_matrix()
    extr = np.zeros((3,3), dtype=np.float32)
    extr[:,:2] = rot[:,:2]
    extr[:,2] = np.squeeze(tvecs[0])
    cv2_H0 = np.dot(intr,extr)
    cv2_H0 /= cv2_H0[2,2]

    wp01 = wps[0][1].copy() # a second point in the first image.
    wp01[2] = 1 # conversion to homogeneous coordinates.

    wp11 = wps[1][1].copy() # a second point in the second image.
    wp11[2] = 1 # conversion to homogeneous coordinates.

    wp21 = wps[2][1].copy() # a second point in the third image.
    wp21[2] = 1 # conversion to homogeneous coordinates.

    opt_ip01 = np.dot(opt_H0, wp01) # an estimated image point of wp01.
    opt_ip11 = np.dot(opt_H1, wp11) # an estimated image point of wp11.
    opt_ip21 = np.dot(opt_H2, wp21) # an estimated image point of wp21.

    cv2_ip01 = np.dot(cv2_H0, wp01) # an estimated image point of wp01.
    cv2_ip11 = np.dot(cv2_H1, wp11) # an estimated image point of wp11.
    cv2_ip21 = np.dot(cv2_H2, wp21) # an estimated image point of wp21.

    cv2_E01 = np.linalg.norm(ips[0][1]-(cv2_ip01/cv2_ip01[2])[:-1])
    cv2_E11 = np.linalg.norm(ips[1][1]-(cv2_ip11/cv2_ip11[2])[:-1])
    cv2_E21 = np.linalg.norm(ips[2][1]-(cv2_ip21/cv2_ip21[2])[:-1])

    opt_E01 = np.linalg.norm(ips[0][1]-(opt_ip01/opt_ip01[2])[:-1])
    opt_E11 = np.linalg.norm(ips[1][1]-(opt_ip11/opt_ip11[2])[:-1])
    opt_E21 = np.linalg.norm(ips[2][1]-(opt_ip21/opt_ip21[2])[:-1])

    print(cv2_E01, opt_E01)
    print(cv2_E11, opt_E11)
    print(cv2_E21, opt_E21)

    print("dist : \n")
    print(dist)
    print("rvecs : \n")
    print(rvecs)
    print("tvecs : \n")
    print(tvecs)
    """

    cv2.destroyAllWindows()

sys.exit()
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