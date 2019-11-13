import glob
import os
import cv2
import imutils


def check_if_dir_exists(path):
    if not os.path.isdir(path):
        os.makedirs(path)


clahe_adjust = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
in_path = "/home/ivan/experiments/sources/clf_test/night/sc_2_parking_g_01/src/"
out_path = "/home/ivan/experiments/sources/clf_test/night/sc_2_parking_g_01/src_320x240_grayscale/"

check_if_dir_exists(out_path)

img_paths = glob.glob(os.path.join(in_path, '*.jpeg'))

i = 0
for img_path in img_paths:
    img = cv2.imread(img_path, 0)
    # img = clahe_adjust.apply(img)
    img = imutils.rotate(img, 2.5)

    # name = os.path.split(img_path)[1][4:-5].zfill(4) + '.jpeg'
    name = os.path.split(img_path)[1]

    out_dir = os.path.join(out_path, name)
    cv2.imwrite(out_dir, img)






