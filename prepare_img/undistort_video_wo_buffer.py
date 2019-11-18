import cv2
import glob
import os
import numpy as np
import subprocess as sp
import logging


# in_dir = 'raw_video/'
# out_dir = 'raw_video/res_img/'

ffmpeg_bin = r'ffmpeg'


in_dir = '/home/ivan/experiments/sources/clf_test/night/v/'
out_dir = '/home/ivan/experiments/sources/clf_test/night/v/'

vid_paths = glob.glob(os.path.join(in_dir, '*.mp4'))

mtx = np.array([[1.33528711e+03, 0.00000000e+00, 5.95825150e+02],
                [0.00000000e+00, 1.33790643e+03, 3.57672066e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dst = np.array([[-3.99390135e-01, -2.87038173e-01, -1.87622642e-03,  2.57836298e-03, 1.95568750e+00]])

try:
    for vid_path in vid_paths:
        cap = cv2.VideoCapture(vid_path)
        logging.info('{} is processing'.format(vid_path))

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(5))

        name = os.path.split(vid_path)[1]
        out_path = os.path.join(out_dir, '{}_{}'.format('corrected', name))

        command = [ffmpeg_bin,
                   '-y',
                   '-f', 'rawvideo',
                   '-vcodec', 'rawvideo',
                   '-s', '{}x{}'.format(frame_width, frame_height),
                   '-pix_fmt', 'bgr24',
                   '-r', '{}'.format(fps),
                   '-i', '-',
                   '-an',
                   '-vcodec', 'libx264',
                   out_path]

        proc = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)

        while cap.isOpened():
            ret, img = cap.read()
            if ret:
                img = cv2.undistort(img, mtx, dst)
                proc.stdin.write(img.tostring())
            else:
                proc.stdin.close()
                proc.stderr.close()
                proc.wait()

                break

except KeyboardInterrupt:
    pass



