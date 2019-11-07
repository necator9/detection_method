import cv2
import glob
import os
import numpy as np

# in_dir = 'raw_video/'
# out_dir = 'raw_video/res_img/'

ffmpeg_bin = r'ffmpeg'


in_dir = '/home/ivan/ip_cam/res'
out_dir = '/home/ivan/ip_cam/res_img'

vid_paths = glob.glob(os.path.join(in_dir, '*.mp4'))

mtx = np.array([[1.33528711e+03, 0.00000000e+00, 5.95825150e+02],
                [0.00000000e+00, 1.33790643e+03, 3.57672066e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dst = np.array([[-3.99390135e-01, -2.87038173e-01, -1.87622642e-03,  2.57836298e-03, 1.95568750e+00]])

try:
    for vid_path in vid_paths:
        data = []
        cap = cv2.VideoCapture(vid_path)

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(5))

        name = os.path.split(vid_path)[1][:-4]
        while cap.isOpened():
            ret, img = cap.read()
            img = cv2.undistort(img, mtx, dst)
            if ret:
                data.append(img)
                # if len(data) == 1000:
                #     write_images(data, out_dir)
                #     break
            else:
                start = data[-200:]
                start.reverse()
                end = data[:-200]
                dt = start + end
                for i, img in enumerate(dt):
                    out_path = os.path.join(out_dir, '{}_{}.jpeg'.format(name, i))
                    cv2.imwrite(out_path, img)

except KeyboardInterrupt:
    pass


command = [ffmpeg_bin,
           '-y',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-s', '{}x{}'.format(frame_width, frame_height),
           '-pix_fmt', 'bgr24',
           # '-pix_fmt', 'gray',

           '-r', '{}'.format(fps),
           '-i', '-',
           '-an',
           '-vcodec', 'libx264',

           os.path.join(out_dir, 'movement_{}_{}_{}.mp4'.format(img_number, COUNTER, tm))]
