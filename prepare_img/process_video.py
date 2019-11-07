import cv2
import glob
import os
import logging
import numpy as np
import pandas as pd
import pinhole_camera_model as pcm
import imutils

logging.basicConfig(level=logging.INFO)

in_path = '/home/ivan/movement/pedestrian/'


def find_obj_params(mog_mask, height, pinhole_cam):
    _, contours, _ = cv2.findContours(mog_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c_areas = [cv2.contourArea(cnt) for cnt in contours]
    polygon_idx = c_areas.index(max(c_areas))
    polygon = contours[polygon_idx]
    b_r = x, y, w, h = cv2.boundingRect(polygon)
    c_a_px = cv2.contourArea(polygon)

    d = pinhole_cam.pixels_to_distance(height, y + h)
    h_rw = pinhole_cam.get_height(height, d, b_r)
    w_rw = pinhole_cam.get_width(height, d, b_r)

    rect_area_rw = w_rw * h_rw
    rect_area_px = w * h
    extent = float(c_a_px) / rect_area_px
    c_a_rw = c_a_px * rect_area_rw / rect_area_px

    return [d, c_a_rw, w_rw, h_rw, extent, x, y, w, h, c_a_px]


mtx = np.array([[1.33528711e+03, 0.00000000e+00, 5.95825150e+02],
                [0.00000000e+00, 1.33790643e+03, 3.57672066e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dst = np.array([[-3.99390135e-01, -2.87038173e-01, -1.87622642e-03,  2.57836298e-03, 1.95568750e+00]])

f_paths = glob.glob(os.path.join(in_path, '*.mp4'))

camera_height = -4.982
rw_angle = -16.4801139558
f_l = 3.6
w_ccd = 3.4509432207429906
h_ccd = 1.937355215491415
frame_width, frame_height = (1280, 720)

params = []

try:
    for path in f_paths:
        cap = cv2.VideoCapture(path)
        fps = int(cap.get(5))

        pinhole_cam = pcm.PinholeCameraModel(rw_angle=rw_angle, f_l=f_l, w_ccd=w_ccd,
                                             h_ccd=h_ccd, img_res=[frame_width, frame_height])

        logging.info('{}x{}, {} FPS, {}'.format(frame_width, frame_height, fps, os.path.split(path)[1]))
        frame_counter = 0

        while cap.isOpened():
            ret, img = cap.read()
            if ret:
                frame_counter += 1

                img = cv2.undistort(img, mtx, dst)
                img = imutils.rotate(img, 4.7)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.erode(img, None, iterations=1)

                try:
                    vid_params = find_obj_params(img, camera_height, pinhole_cam)
                    time_s = round(frame_counter / float(fps), 2)
                    name = os.path.split(path)[1]
                    params.append(vid_params + [time_s, name])

                except ValueError:
                    continue

            else:
                cap.release()

    raise KeyboardInterrupt

except KeyboardInterrupt:
    out_file = 'h_{}_a_{}_fl_{}_wccd_{}_hccd{}_{}x{}.csv'.format(camera_height, rw_angle, f_l, w_ccd, h_ccd,
                                                                 frame_width, frame_height)
    df_data = pd.DataFrame(params, columns=['d', 'c_a_rw', 'w_rw', 'h_rw', 'extent', 'x',
                                            'y', 'w', 'h', 'c_a_px', 'time', 'name'])
    df_data.to_csv(out_file)


