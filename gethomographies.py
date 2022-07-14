import os, sys, time
from pickletools import optimize
import cv2
from cv2 import checkRange
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import scipy.optimize as spo

"""The homography matrix is calculated.
The 3D world points are given with Z=0.
The coordinates of each image point is 
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
    if os.path.exists(save_dir) is False: os.makedirs(save_dir)

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
            print(f"The corner of {fname} is found.")

        else: 
            #raise ValueError(f"The corner of {fname} is not found.")
            print(f"The corner of {fname} is not found.")

        continue

    (h, w) = gray.shape[::-1]

    return Mwps, Mips, fnames_found, (h,w)

def calibrate(Mwps, Mips):
    """Camera Calibration algorithm by Zhang.

    Parameters
    ----------
    Mwps: a list
        A list with M elements. Each element are a 2D array.
        Each row in an 2D array represents the world coordinate, i.e., (X,Y,Z) of a corner.
        There are M images and each has N corners.
        Thus, len(Mwps)=M and Mwps[k].shape = (N,3) where k is an integer 0<k<N-1.

    Mips: a list
        The associated observed coordinate of each corners, i.e., (u,v) of a corner, in the normalized image frame.
        len(Mips)=M and Mips[k].shape = (N,2) where k is an integer 0<k<N-1.

    Returns
    -------
    opt_intr: ndarray
        optimized intrinsic parameters.

    opt_rdist: tuple
        optimized radial distortion parameters.

    opt_Mexpr: list
        optimized extrinsic parameters for each M images.
    """

    MH = get_homographies(Mwps, Mips) # M homography matrice.
    intr = get_camera_intrinsic(MH, gamma=1) # intrinsic paramters.
    Mextr, Mrvec, Mtvecs, Mextr_z0  = get_extrinsics(intr, MH) # M extrinsic parameters for associated M images.
    rdist = estimate_lens_distortion(intr, Mextr, Mwps, Mips) # radial distortion parameters, i.e., (k0, k1).

    opt_intr, opt_rdist, opt_Mexpr = refine_all(intr, rdist, Mextr, Mwps, Mips)

    return opt_intr, opt_rdist, opt_Mexpr

def get_homographies(Mwps, Mips):

    M = len(Mwps) # The number of the views, i.e. the images.
    Hs = []

    for m, (wps, ips) in enumerate(zip(Mwps,Mips)):

        H_init = estimate_homography(wps, ips)
        opt_H, jac = refine_homography(H_init, wps, ips)
        Hs.append(opt_H)

    return Hs

def estimate_homography(wps, ips):
    """Estimate the homography matrix associated with each image.

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

    result = spo.least_squares(residuals_h, h, method='lm', args=(M, wps))
    #result = spo.leastsq(residuals, h, args=(M, wps)) # MINPACK wrapper.

    # return the optimized homography.
    hh = result.x
    #hh = result[0]
    hh /= hh[8]
    hh = hh.reshape(3,3)

    J = result.jac

    return hh, J

def residuals_h(h, M, wps):

    Y = value_h(wps, h) # the estimated corner locations for an image, in sensor coordinate.
    E = M - Y # a function to minimize.
    
    return E

def value_h(wps, h):
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

def jac_h(h, wps):
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

def get_camera_intrinsic(MH, gamma=0):
    """Using Zhang's technique, obtain the intrinsic parameters of a camera.

    Parameters
    ----------
    MH: list
        a list of homographies.

    Returns
    -------
    A: ndarray
        A 3x3 ndarray.
    """

    M = len(MH) # The number of images.
    A = np.zeros((3,3), dtype=np.float64)
    V = np.zeros((2*M,6), dtype=np.float64)

    for m, H in enumerate(MH):
        V[2*m  ] = vpq(H,0,1)
        V[2*m+1] = vpq(H,0,0) - vpq(H,1,1)

    U, S, Vh = np.linalg.svd(V)
    b = Vh[-1,:]

    (B0, B1, B2, B3, B4, B5) = b
    if gamma == 0: B1 = 0

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

def get_extrinsics(intr, MH):
    """Get extrinsic parameters by using the intrinsic parameters and homographies.

    Parameters
    ----------
    intr: ndarray
        a 3x3 intrinsic parameters.

    MH: list
        a list of the homography matrice.

    Returns
    -------
    Ws: list

    Rs: list

    Ts: list

    Ws_z0: list
        a list of 3x3 ndarrays. Each ndarray has the form of (r1, r2, t),
        where r1,r2 and t are the column vectors.
        In Zhang's technique, since the world coordinate of the corners are fixed to z=0,
        r3 vectors are usually omitted.
    """

    Ws = []
    Rs = []
    Ts = []
    Ws_z0 = []
    W = np.zeros((3,4), dtype=np.float64)

    for m, H in enumerate(MH):

        W, R, T, W_z0 = estimate_view_transform(intr,H)
        Ws.append(W)
        Rs.append(R)
        Ts.append(T)
        Ws_z0.append(W_z0)

    return Ws, Rs, Ts, Ws_z0

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
    W_z0 = np.concatenate((R[:,:2],T[:,None]), axis=1)

    return W, R, T, W_z0

def estimate_lens_distortion(intr, Mextr, Mwps, Mips):

    M = len(Mwps) # The number of views.
    N = Mwps[0].shape[0] # The number of points.

    uc = intr[0,2]
    vc = intr[1,2]

    D = np.zeros((2*M*N, 2), dtype=np.float64)
    d = np.zeros(2*M*N, dtype=np.float64)
    l = 0

    for m, (wps, ips) in enumerate(zip(Mwps,Mips)):

        for n, (wp, ip) in enumerate(zip(wps, ips)):

            ansx, ansy = image_to_normalized_projection(Mextr[m], wp)
            r = np.sqrt(ansx**2 + ansy**2)
            u, v = normalized_to_sensor_projection(intr, (ansx, ansy))
            du, dv = u-uc, v-vc
            D[2*l  ] = (du*(r**2), du*(r**4))
            D[2*l+1] = (dv*(r**2), dv*(r**4))
            udot = ip[0]
            vdot = ip[1]
            d[2*l  ] = udot - u
            d[2*l+1] = vdot - v
            l += 1

    sol = spo.lsq_linear(D, d, lsq_solver='exact')
    #sol = np.linalg.multi_dot([np.linalg.inv(np.dot(D.T, D)), D.T, d])

    return sol.x

def image_to_normalized_projection(extr, wp):

    hom_wp = np.ones(4, dtype=np.float64)
    hom_wp[:3] = wp

    vec = np.dot(extr, hom_wp)
    vec /= vec[2]

    (x, y) = vec[0:2]

    return (x,y)

def normalized_to_sensor_projection(intr, normcoor):

    hom_normcoor = np.ones(3, dtype=np.float64)
    hom_normcoor[:2] = normcoor

    (u,v,z) = np.dot(intr, hom_normcoor)

    return (u,v)

def refine_all(intr, rdist, Mextr, Mwps, Mips):

    P_init = compose_parameter_vector(intr, rdist, Mextr)

    M = len(Mextr)
    N = len(Mwps[0])
    Y = np.zeros(2*M*N, dtype=np.float64)

    for m, (wps, ips) in enumerate(zip(Mwps,Mips)):
        for n, (wp,ip) in enumerate(zip(wps, ips)):
            Y[N*m+n  ] = ip[0]
            Y[N*m+n+1] = ip[1]

    start = time.time()
    result = spo.least_squares(residuals_P, P_init, method='lm', args=(Mwps, Y))
    end = time.time()
    elapsed_time = end-start
    print(elapsed_time)
    P_opt = result.x
    intr, k, Mextr = decompose_parameter_vector(P_opt)

    return intr, k, Mextr

def residuals_P(P, Mwps, Y_dist_observed):

    Y_dist_model = value_P(Mwps, P)
    E = Y_dist_model - Y_dist_observed

    return E

def value_P(Mwps, P):

    M = len(Mwps) # the number of images.
    N = Mwps[0].shape[0] # the number of corners.
    Y_dist_model = np.zeros(2*M*N, dtype=np.float64)
    intr = np.zeros((3,3), dtype=np.float64)

    intr[0,0] = P[0] # alpha
    intr[1,1] = P[1] # beta
    intr[0,1] = P[2] # gamma
    intr[0,2] = P[3] # uc
    intr[1,2] = P[4] # vc
    intr[2,2] = 1

    k0 = P[5] # k0
    k1 = P[6] # k1

    for m, wps in enumerate(Mwps):
        i = 6*m+7
        (rhox, rhoy, rhoz, tx, ty, tz) = P[i:i+6]
        rho = np.array((rhox, rhoy, rhoz), dtype=np.float64)
        t = np.array((tx, ty, tz), dtype=np.float64)
        R = to_rotation_matrix(rho)
        extr = np.concatenate((R,t[:,None]), axis=1)
        for n, wp in enumerate(wps):
            (projx, projy) = image_to_normalized_projection(extr, wp)
            (x_warp, y_warp) = warp((projx, projy), (k0, k1))
            (u, v) = normalized_to_sensor_projection(intr, (x_warp,y_warp))

            Y_dist_model[2*N*m+2*n  ] = u
            Y_dist_model[2*N*m+2*n+1] = v

    return Y_dist_model

def warp(projp, rdist):

    x = projp[0]
    y = projp[1]
    r = np.sqrt(x**2+y**2)
    k0 = rdist[0]
    k1 = rdist[1]

    x_warp = x * (1+k0*r**2+k1*r**4)
    y_warp = y * (1+k0*r**2+k1*r**4)

    return (x_warp, y_warp)

def compose_parameter_vector(intr, rdist, Mextr):

    alpha = intr[0,0]
    beta = intr[1,1]
    gamma = intr[0,1]
    uc = intr[0,2]
    vc = intr[1,2]
    k0 = rdist[0]
    k1 = rdist[1]

    M = len(Mextr)
    P = np.zeros(7+6*M, dtype=np.float64)
    P[0:7] = (alpha, beta, gamma, uc, vc, k0, k1)

    for m, w in enumerate(Mextr):
        R = w[:,:3]
        t = w[:,3]
        rho = to_rodrigues_vector(R)
        P[6*m+7:6*m+13] = (rho[0], rho[1], rho[2], t[0], t[1], t[2])

    return P

def decompose_parameter_vector(P):
    """Decompose the parameter vector P into
    the intrinsic, extrinsic and radial distortion parameters.

    Parameters
    ----------
    P : a tuple.
        A parameter vector, (intr, rdist, extr).

    Returns
    -------
    intr : a tuple.
        (alpha, beta, gamma, uc, vc)

    rdist : a tuple.
        (k0, k1)

    Mextr : a list.
        A list that each element is a tuple. 
        The length of a list is M which is the number of images.
        Each tuple is the extrinsic parameters of the associated view.
        Each tuple has the form of (rho_x, rho_y, rho_z, t_x, t_y, t_z).
    """
    A = np.zeros((3,3), dtype=np.float64)
    A[0,0] = P[0] # alpha
    A[0,1] = P[2] # gamma
    A[0,2] = P[3] # uc
    A[1,1] = P[1] # beta
    A[1,2] = P[4] # vc
    A[2,2] = 1
    k = np.array([P[5], P[6]])
    M = int((len(P)-7)/6)
    Ws = []

    for i in range(M):
        m = 6*i+7
        rho = P[m:m+3]
        t = P[m+3:m+6]
        R = to_rotation_matrix(rho)
        W = np.concatenate((R,t[:,None]), axis=1)
        Ws.append(W)

    return A, k, Ws

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

def to_rodrigues_vector(R):
    """Conversion from a 3D rotation matrix to the associated Rodrigues vector."""

    p = 0.5 * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    c = 0.5 * (np.trace(R)-1)

    if np.linalg.norm(p) == 0:
        if c == 1: rho = np.zeros(3, dtype=np.float64)
        elif c == -1:
            Rp = R + np.identity(3)
            norm_col = np.linalg.norm(Rp, axis=0)
            arg_max_norm_col = np.argmax(norm_col)
            v = Rp[:,arg_max_norm_col]
            u = v / norm_col[arg_max_norm_col]
            (u0, u1, u2) = u
            if (u0 < 0) or ((u0==0) and (u1<0)) or ((u0==u1==0) or (u2<0)): u = -u
            rho = np.pi * u
        else: raise ValueError("Rodrigues vector cannot be drived due to the wrong rotation matrix.")

    else:
        normp = np.linalg.norm(p)
        u = p / normp
        theta = np.arctan(normp/c)
        rho = theta * u

    return rho

def draw_corners(raw_dir, save_dir, checkerboardhw):

    fnames_raw = os.listdir(raw_dir)
    compact_corners = {}

    for fname in fnames_raw:

        img = cv2.imread(raw_dir+fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCornersSB(gray, checkerboardhw, \
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK\
            + cv2.CALIB_CB_NORMALIZE_IMAGE)

        compact_corners[fname] = np.squeeze(corners)

        img = cv2.drawChessboardCorners(img, checkerboardhw, corners, ret)
        cv2.imwrite(f'{save_dir}{fname}_draw_corners.png', img)

    return compact_corners

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
            undist_dir = './images_undistorted_draw_corners/'
            img = cv2.drawChessboardCorners(dst, CHECKERBOARD, corners, ret)
            cv2.imwrite(f'{undist_dir}{fname}', img)

    return undist_corners

if __name__ == '__main__':

    # Defining the dimensions of checkerboard.
    w = 9
    h = 6
    CHECKERBOARD = (h, w)
    raw_dir = './images_raw_small/'
    save_dir = './images_undistorted_small/'

    Mwps, Mips, fnames, (h_npixels, w_npixels) = get_world_points(CHECKERBOARD, raw_dir, save_dir)

    ret, cv2_intr, cv2_dist, cv2_rvecs, cv2_tvecs = cv2.calibrateCamera(Mwps, Mips, (h_npixels, w_npixels), None, None)

    cv2_rot, cv2_jac = cv2.Rodrigues(cv2_rvecs[0])

    opt_intr, opt_rdist, opt_Mexpr = calibrate(Mwps, Mips)

    """
    # Homography part is completely developed.
    undist_corners = get_undistorted_corners(raw_dir, CHECKERBOARD, cv2_intr, cv2_dist, save=True)
    Errs0, cv2_ip0, opt_ip0, cv2_H0, opt_H0 = homography_compare(Mwps[0], Mips[0], undist_corners['image_10.jpg'])
    Errs1, cv2_ip1, opt_ip1, cv2_H1, opt_H1 = homography_compare(Mwps[1], Mips[1], undist_corners['image_11.jpg'])

    A = get_camera_intrinsic(Hs, gamma=True)
    Ws, Rs, Ts, hom_Ws = get_extrinsics(A, Hs)

    k = estimate_lens_distortion(A, Ws, Mwps, Mips)
    opt_intr, opt_k, opt_Mextr = refine_all(A, k, Ws, Mwps, Mips)

    recon_H = np.dot(A,hom_Ws[0])
    recon_H /= recon_H[2,2]
    
    df = pd.DataFrame(Errs0, columns=['opencv', 'mine'])
    ax = df.plot(style=['-', 'o'])
    ax.set_xlabel('Corner number')
    ax.set_ylabel(r'$\|\|e\|\|^2$')
    ax.get_figure().savefig('error.png', bbox_inches='tight')

    print("dist : \n")
    print(dist)
    print("rvecs : \n")
    print(rvecs)
    print("tvecs : \n")
    print(tvecs)
    """

    cv2.destroyAllWindows()