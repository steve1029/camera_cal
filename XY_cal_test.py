import os, sys, time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gethomographies as gh

h = 5
w = 13

checkerboardhw = (h,w)

raw_dir = './XY_cal_test_image/'
save_dir = './XY_cal_test_image_result/'

Mwps, Mips, fnames, (h_npixels, w_npixels) = gh.get_world_points(checkerboardhw, raw_dir, save_dir)
corners = gh.draw_corners(raw_dir, save_dir, checkerboardhw)