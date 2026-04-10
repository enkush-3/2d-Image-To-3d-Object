import cv2 as cv
import matplotlib.pyplot as plt

import glob

imgs = []

folder_path = '/home/enkush-3/Documents/University_3_1/Computer_vision/Biy_Daalt_v0.2/Dataset/'
for img_path in glob.glob(folder_path + '*.jpg'):
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgs.append(gray)


akaze = cv.AKAZE_create()

imgs_key_points = []
imgs_descriptor = []
for img in imgs:
    kp_akaze, des_akaze = akaze.detectAndCompute(img, None)
    imgs_key_points.append(kp_akaze)
    imgs_descriptor.append(des_akaze)