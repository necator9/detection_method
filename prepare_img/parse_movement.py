import subprocess as sp
import cv2
import glob
import os
import sys
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)


lt = int(sys.argv[1])
ut = int(sys.argv[2])

ffmpeg_bin = r'ffmpeg'

dir_path = '/home/ivan/resampled'
out_dir = '/home/ivan/movement1'

dir_path = '/home/imatveev/video_rec_proc'
out_dir = '/home/imatveev/movement'

img_paths = glob.glob(os.path.join(dir_path, '*.mp4'))

img_names_float = sorted([int(os.path.split(img_name)[1][9:-4]) for img_name in img_paths])

img_names_float = [img_number for img_number in img_names_float if lt <= img_number <= ut]


def detect_movement(filter_kernel, img):
    global MOG2
    global FRAME_COUNTER
    global MARGIN
    global frame_height
    global frame_width

    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, filter_kernel)
    img = cv2.blur(img, (9, 9))
    # img = cv2.blur(img, (3, 3))
    # img = cv2.GaussianBlur(img, (5, 5), 0)

    mog_mask = MOG2.apply(img)
    # the_imge = np.copy(mog_mask)
    _, mog_mask = cv2.threshold(mog_mask, 127, 255, cv2.THRESH_BINARY)
    mog_mask = cv2.morphologyEx(mog_mask, cv2.MORPH_OPEN, filter_kernel)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, filter_kernel)

    mog_mask = cv2.dilate(mog_mask, None, iterations=1)
    contours, _ = cv2.findContours(mog_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    FRAME_COUNTER += 1

    c_areas = [cv2.contourArea(cnt) for cnt in contours]

    stats = [True if c_a > THR_VAL else False for c_a in c_areas]

    stat = any(stats)

    # if stat:
    #     b_rs = [cv2.boundingRect(cnt) for cnt in contours]

        # for i, b_r in enumerate(b_rs):
        #     x, y, w, h = b_r
        #     if (y + h > frame_height - MARGIN) or (y < MARGIN) or (x + w > frame_width - MARGIN) or (x < MARGIN):
        #         # print "lol"
        #         stats[i] = False
        #         stat = any(stats)

    return img, stat, mog_mask #filtered_mask


is_proc_opened = False

try:
    for img_number in img_names_float:
        name = os.path.join(dir_path, 'resampled{}.mp4'.format(img_number))
        logging.info('File {} chosen'.format(name))

        cap = cv2.VideoCapture(name)
        logging.debug('VideoCapture created')

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(5))
        logging.info('{}x{}, {} FPS'.format(frame_width, frame_height, fps))

        MOG2 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        FRAME_COUNTER = 0
        f_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        is_proc_opened = False

        THR_VAL = 3000
        COUNTER = 0
        MARGIN = 40

        while cap.isOpened():
            logging.debug('Before first movement detected')
            ret, img = cap.read()
            logging.debug('First movement detected, ret: {}'.format(ret))

            if ret:
                img, status, filtered_mask = detect_movement(f_kernel, img)

                if status:
                    tm = round(FRAME_COUNTER / fps / float(60), 2)
                    print name, tm
                    command = [ffmpeg_bin,
                               '-y',
                               '-f', 'rawvideo',
                               '-vcodec', 'rawvideo',
                               '-s', '{}x{}'.format(frame_width, frame_height),
                               # '-pix_fmt', 'bgr24',
                               '-pix_fmt', 'gray',

                               '-r', '{}'.format(fps),
                               '-i', '-',
                               '-an',
                               '-vcodec', 'libx264',
                               # '-vcodec', 'lib',
                              # '-b', '277777',
                               # '-preset', 'superfast',
                               os.path.join(out_dir, 'movement_{}_{}_{}.mp4'.format(img_number, COUNTER, tm))]

                    proc = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)
                    is_proc_opened = True

                    proc.stdin.write(filtered_mask.tostring())

                    it = 0
                    while it < 200:
                        logging.debug('Before second movement detected')
                        ret, img = cap.read()
                        logging.debug('Second movement detected')

                        if ret:
                            img, status, filtered_mask = detect_movement(f_kernel, img)

                            proc.stdin.write(filtered_mask.tostring())  # img

                            it += 1

                            if status:
                                it = 0
                        else:
                            break

                    COUNTER += 1

                    proc.stdin.close()
                    proc.stderr.close()
                    proc.wait()

            else:
                break
except KeyboardInterrupt:
    if is_proc_opened:
        proc.stdin.close()
        proc.stderr.close()
        proc.wait()

