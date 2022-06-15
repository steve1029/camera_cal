import cv2
import numpy as np
from glob import glob

CHECKERBOARD = (6, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

print(cv2.TERM_CRITERIA_EPS)
print(cv2.TERM_CRITERIA_MAX_ITER)
print(cv2.TERM_CRITERIA_COUNT)
print(criteria)

imgfnames = glob('./images/*.jpg')

print(imgfnames)

"""
CALIB_CB_ADAPTIVE_THRESH: Use adaptive thresholding to convert the image to black and white, rather than a fixed threshold level (computed from the average image brightness).
CALIB_CB_NORMALIZE_IMAGE: Normalize the image gamma with equalizeHist before applying fixed or adaptive thresholding.
CALIB_CB_FILTER_QUADS: Use additional criteria (like contour area, perimeter, square-like shape) to filter out false quads extracted at the contour retrieval stage.
CALIB_CB_FAST_CHECK: Run a fast check on the image that looks for chessboard corners, and shortcut the call if none is found. This can drastically speed up the call in the degenerate condition when no chessboard is observed.
"""

print(cv2.CALIB_CB_ADAPTIVE_THRESH)
print(cv2.CALIB_CB_FAST_CHECK)
print(cv2.CALIB_CB_NORMALIZE_IMAGE)