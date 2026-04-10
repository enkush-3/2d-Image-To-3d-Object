import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('/home/enkush-3/Documents/University_3_1/Computer_vision/Biy_Daalt_v0.2/Dataset/IMG_20230210_154338_jpg.rf.ea7cf1b875e67b6f0945a18a5445fb56.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

akaze = cv.AKAZE_create()
kp_akaze, des_akaze = akaze.detectAndCompute(gray, None)
img_akaze = cv.drawKeypoints(img, kp_akaze, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

print(des_akaze.shape)

orb = cv.ORB_create()
kp_orb, des_orb = orb.detectAndCompute(gray, None)
img_orb = cv.drawKeypoints(img, kp_orb, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

sift = cv.SIFT_create()
kp_sift, des_sift = sift.detectAndCompute(gray, None)
img_sift = cv.drawKeypoints(img, kp_sift, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure(figsize=(15,10))
plt.subplot(1,3,1), plt.imshow(cv.cvtColor(img_akaze, cv.COLOR_BGR2RGB)), plt.title('AKAZE')
plt.subplot(1,3,2), plt.imshow(cv.cvtColor(img_sift, cv.COLOR_BGR2RGB)), plt.title('SIFT')
plt.subplot(1,3,3), plt.imshow(cv.cvtColor(img_orb, cv.COLOR_BGR2RGB)), plt.title('ORB')

plt.show()