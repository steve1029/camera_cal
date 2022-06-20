import os, sys
from pickletools import optimize
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    Mwps = []
    wps = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    wps[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) 

    #print(wps)

    # Creating vector to store vectors of the 2D sensor points for each checkerboard image.
    # Note that this vector lies in the sensor coordinates, not the image plane.
    Mips = [] # findChessboardCornersSB()

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
            Mwps.append(wps)
            #Mips.append(corners)
            Mips.append(np.squeeze(corners))

        else: 
            #raise ValueError(f"The corner of {fname} is not found.")
            print(f"The corner of {fname} is not found.")
            continue

    (h, w) = gray.shape[::-1]

    return Mwps, Mips, fnames_found, (h,w)

def get_homographies(Mwps, Mips):

    M = len(Mwps) # The number of the views, i.e. the images.
    Hs = []

    for m, (wps, ips) in enumerate(zip(Mwps,Mips)):

        H_init = estimate_homography(wps, ips)
        opt_H, jac = refine_homography(H_init, wps, ips)
        Hs.append(opt_H)

    return Hs

def estimate_homography(wps, ips):
    """Estimate the homography matrix for each image

    Parameters
    ----------
    Mwps: a list.
        a list of 2D arrays. a world points of the chessboard corners.
        Note that in Zhang's technique, Z is omitted for the world coordinates
        because Z=0 for all chessboard corners.

    Mips: a list.
        a list of 2D vectors, an observed sensor points in an images.

    Returns
    -------
    H: ndarray.
        3x3 ndarray. A homography matrix of an image.
    """

    N = wps.shape[0] # The number of points in an image.
    Np = get_normalization_matrix(wps)
    Nq = get_normalization_matrix(ips)
    M = np.zeros((2*N,9), dtype=np.float32)

    homwp = wps.copy()
    homip = ips.copy()

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

def refine_homography(H, wps, ips):
    """
    Returns
    -------
    optH: ndarray
        3x3 homography matrix. Numerically optimized.
    """

    N = wps.shape[0] # The number of corners.

    M = np.zeros(2*N, dtype=np.float64) # observed corner locations, in sensor coordinate.

    for j, q in enumerate(ips):

        M[2*j  ] = q[0]
        M[2*j+1] = q[1]

    h = H.reshape(-1)

    result = spo.least_squares(residuals, h, method='lm', args=(M, wps))
    #result = spo.leastsq(residuals, h, args=(M, wps)) # MINPACK wrapper.

    # return the optimized homography.
    hh = result.x
    #hh = result[0]
    hh /= hh[8]
    hh = hh.reshape(3,3)

    J = result.jac

    return hh, J

def residuals(h, M, wps):

    Y = value(wps, h) # the estimated corner locations for an image, in sensor coordinate.
    E = M - Y # a function to minimize.
    
    return E

def value(wps, h):
    """Estimate the coordinate of the image points that lies in the sensor coordinate.

    Parameters
    ----------
    wps: ndarray.
        An 2D array containing an 3D vector coordinate in each row. 
        ex) wps[j] = np.array([x,y,z]).

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

    N = wps.shape[0] # The number of corners.
    Y = np.zeros(2*N, np.float64)

    for j in range(N):

        # (x,y,z) is the world coordinate.
        (x,y,z) = wps[j]

        w = h[6]*x + h[7]*y + h[8]
        u = (h[0]*x + h[1]*y + h[2]) / w
        v = (h[3]*x + h[4]*y + h[5]) / w

        Y[2*j  ] = u
        Y[2*j+1] = v

    return Y

def jac(h, wps):
    """Get Jacobian matrix of size 2N x 9 where N is the number of corners.
    
    Parameters
    ----------
    wps: ndarray.
        An 2D array containing an 3D vector coordinate in each row. 
        ex) wps[j] = np.array([x,y,z]).

    h: ndarray.
        A flattened homography matrix.
    """

    N = wps.shape[0] # The number of corners.
    J = np.zeros((2*N,9), np.float64)

    for j, p in enumerate(wps):
        (X,Y,Z) = p
        sx = h[0]*X + h[1]*Y + h[2]
        sy = h[3]*X + h[4]*Y + h[5]
        w  = h[6]*X + h[7]*Y + h[8]

        J[2*j  ] = [X/w, Y/w, 1/w, 0, 0, 0, -sx*X/w**2, -sx*Y/w**2, -sx/w**2]
        J[2*j+1] = [0, 0, 0, X/w, Y/w, 1/w, -sy*X/w**2, -sy*Y/w**2, -sy/w**2]

    return J

def homography_compare(wps, ips, answer):

    # The homography obtained by using built-in function in OpenCV.
    cv2_H, status = cv2.findHomography(wps, ips)

    # The homography optained by me.
    H_init = estimate_homography(wps, ips)
    opt_H, J = refine_homography(H_init, wps, ips)

    N = wps.shape[0] # The number of corners.

    Errs = np.zeros((N,2), dtype=np.float64)
    cv2_ip = np.zeros((N,2), dtype=np.float64)
    opt_ip = np.zeros((N,2), dtype=np.float64)

    for n, (src, dst) in enumerate(zip(wps,ips)):

        hom_src = src.copy() # a second point in the first image.
        hom_src[2] = 1 # conversion to homogeneous coordinates.

        cv2_dst = np.dot(cv2_H, hom_src) # an estimated image point of wp01.
        opt_dst = np.dot(opt_H, hom_src) # an estimated image point of wp01.

        cv2_E = np.linalg.norm(answer[n]-(cv2_dst/cv2_dst[2])[:-1])
        opt_E = np.linalg.norm(answer[n]-(opt_dst/opt_dst[2])[:-1])

        cv2_ip[n] = (cv2_dst/cv2_dst[2])[:-1]
        opt_ip[n] = (opt_dst/opt_dst[2])[:-1]

        Errs[n] = (cv2_E, opt_E)

    return Errs, cv2_ip, opt_ip, cv2_H, opt_H

def get_camera_intrinsic(Hs, gamma=True):
    """Using Zhang's technique, obtain the intrinsic parameters of a camera.

    Parameters
    ----------

    Returns
    -------
    A: ndarray
        A 3x3 ndarray.
    """

    M = len(Hs) # The number of images.
    A = np.zeros((3,3), dtype=np.float64)
    V = np.zeros((2*M,6), dtype=np.float64)
    for m, H in enumerate(Hs):

        V[2*m  ] = vpq(H,0,1)
        V[2*m+1] = vpq(H,0,0) - vpq(H,1,1)

    U, S, Vh = np.linalg.svd(V)
    b = Vh[np.argmin(S)]

    (B0, B1, B2, B3, B4, B5) = b
    if gamma is False: B1 = 0

    w = B0*B2*B5 - B5*B1**2 - B0*B4**2 + 2*B1*B3*B4 - B2*B3**2
    d = B0*B2 - B1**2
    alpha = np.sqrt(w/(d*B0))
    beta = np.sqrt(w*B0/(d**2))
    gamma = B1*np.sqrt(w/(B0*d**2))
    uc = (B1*B4 - B2*B3)/d
    vc = (B1*B3 - B0*B4)/d

    A[0,0] = alpha
    A[0,1] = gamma
    A[0,2] = uc
    A[1,1] = beta
    A[1,2] = vc
    A[2,2] = 1

    return A

def vpq(H, p, q):

    v = np.zeros(6, dtype=np.float64)
    v[0] = H[0,p]*H[0,q]
    v[1] = H[0,p]*H[1,q] + H[1,p]*H[0,q]
    v[2] = H[1,p]*H[1,q]
    v[3] = H[2,p]*H[0,q] + H[0,p]*H[2,q]
    v[4] = H[2,p]*H[1,q] + H[1,p]*H[2,q]
    v[5] = H[2,p]*H[2,q]

    return v

def get_extrinsics(A, Hs):

    M = len(Hs) # The number of images.
    Ws = []
    Rs = []
    Ts = []
    hom_Ws = []
    W = np.zeros((3,4), dtype=np.float64)

    for m, H in enumerate(Hs):

        W, R, T, hom_W = estimate_view_transform(A,H)
        Ws.append(W)
        Rs.append(R)
        Ts.append(T)
        hom_Ws.append(hom_W)

    return Ws, Rs, Ts, hom_Ws

def estimate_view_transform(A,H):

    h0 = H[:,0] 
    h1 = H[:,1] 
    h2 = H[:,2] 
    
    Q = np.zeros((3,3), dtype=np.float64)

    invA = np.linalg.inv(A)
    invAh0 = np.dot(invA, h0)
    invAh1 = np.dot(invA, h1)
    invAh2 = np.dot(invA, h2)

    kappa = 1/np.linalg.norm(invAh0)
    r0 = kappa * invAh0
    r1 = kappa * invAh1
    r2 = np.cross(r0, r1)
    T = kappa * invAh2

    Q[:,0] = r0
    Q[:,1] = r1
    Q[:,2] = r2

    U, S ,Vh = np.linalg.svd(Q)
    R = np.dot(U, Vh)

    W = np.concatenate((R,T[:,None]), axis=1)
    hom_W = np.concatenate((R[:,:2],T[:,None]), axis=1)

    return W, R, T, hom_W

def to_rotation_matrix(rho):

    rho = np.squeeze(rho)
    theta = np.linalg.norm(rho)
    rhohat = rho / theta

    W = np.zeros((3,3), dtype=np.float64)
    W[0] = (0, -rhohat[0], rhohat[1])
    W[1] = (rhohat[2], 0, -rhohat[0])
    W[2] = (-rhohat[1], rhohat[0], 0)

    R = np.identity(3) + W*np.sin(theta) + np.dot(W, W) * (1-np.cos(theta))

    return R

def get_undistorted_corners(raw_dir, CHECKERBOARD, intr, dist, save=False):

    h, w = CHECKERBOARD
    undist_corners = {} 
    fnames_raw = os.listdir(raw_dir)

    # Using the derived camera parameters to undistort the image
    for fname in fnames_raw:

        img = cv2.imread(raw_dir+fname)
        # Refining the camera matrix using parameters obtained by calibration
        #newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intr, dist, (w, h), 1, (w, h))

        # Method 1 to undistort the image
        #dst = cv2.undistort(img, intr, dist, None, newcameramtx)
        dst = cv2.undistort(img, intr, dist, None, intr)

        # Method 2 to undistort the image
        #mapx, mapy = cv2.initUndistortRectifyMap(intr, dist, None, newcameramtx, (w, h), 5)
        #dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

        # Displaying the undistorted image
        #cv2.imshow("undistorted image", dst)
        #cv2.waitKey(0)

        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD, \
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK\
            + cv2.CALIB_CB_NORMALIZE_IMAGE)

        undist_corners[fname] = np.squeeze(corners)
        # Save the undistorted image.
        if save is True: 
            undist_dir = './images_undistorted_corners/'
            img = cv2.drawChessboardCorners(dst, CHECKERBOARD, corners, ret)
            cv2.imwrite(f'{undist_dir}{fname}', img)

    return undist_corners

def estimate_lens_distortion(intr, Mextr, Mwps, Mips):

    M = len(Mwps) # The number of views.
    N = Mwps[0].shape[0] # The number of points.

    uc = intr[0,2]
    vc = intr[1,2]

    D = np.zeros((2*M*N, 2), dtype=np.float64)
    d = np.zeros(2*M*N, dtype=np.float64)
    l = 0

    for m, (wps, ips) in enumerate(zip(Mwps,Mips)):

        for j in range(N):

            ansx, ansy = normalized_projection(Mextr[m], wps[j])
            r = np.sqrt(ansx**2 + ansy**2)
            u, v = normalized_to_sensor_projection(intr, (ansx, ansy))
            du, dv = u-uc, v-vc
            D[2*l]

    return

def normalized_projection(W, wp):

    hom_wp = np.ones(4, dtype=np.float64)
    hom_wp[0:4] = wp

    vec = np.dot(W, hom_wp)
    vec /= vec[2]

    (x, y) = vec[0:2]

    return (x,y)

def normalized_to_sensor_projection(intr, normcoor):

    hom_normcoor = np.zeros(3, dtype=np.float64)
    hom_normcoor[:2] = normcoor

    (u,v) = np.dot(intr[:2], hom_normcoor)

    return (u,v)

if __name__ == '__main__':

    # Defining the dimensions of checkerboard.
    w = 9
    h = 6
    CHECKERBOARD = (h, w)
    raw_dir = './images_raw/'
    save_dir = './strategy3/'

    Mwps, Mips, fnames, (h_npixels, w_npixels) = get_world_points(CHECKERBOARD, raw_dir, save_dir)

    Hs = get_homographies(Mwps, Mips)

    #A_gam0 = get_camera_intrinsic(Hs, gamma=False)
    #Ws_gam0, Rs_gam0, Ts_gam0 = get_extrinsics(A_gam0, Hs)
    #print(Rs_gam0[0])

    A = get_camera_intrinsic(Hs, gamma=True)
    Ws, Rs, Ts, hom_Ws = get_extrinsics(A, Hs)
    recon_H = np.dot(A,hom_Ws[0])
    recon_H /= recon_H[2,2]
    
    ret, intr, dist, rvecs, tvecs = cv2.calibrateCamera(Mwps, Mips, (h_npixels, w_npixels), None, None)
    rot_cv, jac_rot = cv2.Rodrigues(rvecs[0])

    undist_corners = get_undistorted_corners(raw_dir, CHECKERBOARD, intr, dist, save=True)

    Errs0, cv2_ip0, opt_ip0, cv2_H0, opt_H0 = homography_compare(Mwps[0], Mips[0], undist_corners['image_10.jpg'])
    Errs1, cv2_ip1, opt_ip1, cv2_H1, opt_H1 = homography_compare(Mwps[1], Mips[1],undist_corners['image_11.jpg'])

    df = pd.DataFrame(Errs0, columns=['opencv', 'mine'])
    ax = df.plot(style=['-', 'o'])
    ax.set_xlabel('Corner number')
    ax.set_ylabel(r'$\|\|e\|\|^2$')
    ax.get_figure().savefig('error.png', bbox_inches='tight')

    """
    print("dist : \n")
    print(dist)
    print("rvecs : \n")
    print(rvecs)
    print("tvecs : \n")
    print(tvecs)
    """

    cv2.destroyAllWindows()