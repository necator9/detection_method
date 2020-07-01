import numpy as np
import cv2
import glob
import os

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

# objp = np.zeros((11*4,3), np.float32)
# objp[:,:2] = np.mgrid[0:11, 0:4].T.reshape(-1,2)

objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

nm = '/home/ivan/chessboard.mp4'
nm = '/home/ivan/Desktop/webcam_callib.mp4'
nm = '/home/ivan/Desktop/webcam_callib_320x240_2.mp4'
nm = '/home/ivan/rpi_grabber/callibration_rpi.mp4'
# nm = '/home/ivan/rpi_grabber/callibration_hd3000.mp4'
nm = '/home/ivan/calibration/'
# cap = cv2.VideoCapture(nm)

i = 0
images = glob.glob(nm + '*.png')
try:
    for image in images:
        img = cv2.imread(image)
        ret = True
    # while cap.isOpened():
    #     ret, img = cap.read()
    #     cv2.imshow('img', img)
    #     cv2.waitKey(500)
    #     print(img.shape)
        if ret:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
            # ret, corners = cv2.findCirclesGrid(gray, (4, 11), None, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)

            # If found, add object points, image points (after refining them)

            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                # Draw and display the corners
                # img = cv2.drawChessboardCorners(img, (4, 11), corners2, ret)
                img = cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)
                # name = input("? ")
                # if name == '1':
                objpoints.append(objp)
                imgpoints.append(corners2)
                #     print('Corners added')
                # if name == 'd':
                #     os.remove(image)
            else:
                os.remove(image)
        else:
            raise KeyboardInterrupt

        i += 1
        print(i)

    raise KeyboardInterrupt

except KeyboardInterrupt:
    if len(objpoints) > 0:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        callib_data = [ret, mtx, dist, rvecs, tvecs]

        with open('calibration_rpi_sum.txt', 'w') as writer:
            for dt in callib_data:
                print(dt)
                writer.write('{} \n'.format(dt))


cv2.destroyAllWindows()

