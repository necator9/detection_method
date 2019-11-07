import subprocess as sp
import cv2
import glob
import os
import logging

logging.basicConfig(level=logging.INFO)

ffmpeg_bin = r'ffmpeg'

# in_dir = '/home/ivan/ip_cam/'
# out_dir = '/home/ivan/ip_cam/res'

in_dir = 'raw_video/'
out_dir = 'raw_video/res/'

img_paths = glob.glob(os.path.join(in_dir, '*.mp4'))


def detect_movement(filter_kernel, img):
    global MOG2
    global FRAME_COUNTER
    global frame_height
    global frame_width
    global total_res

    mog_mask = MOG2.apply(img)
    _, mog_mask = cv2.threshold(mog_mask, 127, 255, cv2.THRESH_BINARY)
    mog_mask = cv2.morphologyEx(mog_mask, cv2.MORPH_OPEN, filter_kernel)
    contours, _ = cv2.findContours(mog_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    FRAME_COUNTER += 1

    c_areas = [cv2.contourArea(cnt) for cnt in contours]

    # if len(c_areas) > 0:
    #     print(max(c_areas))
    #     print(max([c_a / float(total_res) for c_a in c_areas]))

    stats = [True if c_a / float(total_res) > THR_VAL else False for c_a in c_areas]
    stat = any(stats)

    return img, stat, mog_mask


is_proc_opened = False

try:
    for img_path in img_paths:
        img_number = int(os.path.split(img_path)[1].split('.')[0][4:])
        logging.info('{} is processing'.format(img_path))

        cap = cv2.VideoCapture(img_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        total_res = frame_width * frame_height
        fps = int(cap.get(5))
        logging.info('{}x{}, {} FPS'.format(frame_width, frame_height, fps))

        MOG2 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        FRAME_COUNTER = 0
        f_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        is_proc_opened = False
        THR_VAL = 0.0003
        COUNTER = 0

        while cap.isOpened():
            logging.debug('Before first movement detected')
            ret, img = cap.read()
            logging.debug('First movement detected, ret: {}'.format(ret))

            if ret:
                img, status, filtered_mask = detect_movement(f_kernel, img)

                if status:
                    tm = round(FRAME_COUNTER / fps / float(60), 2)
                    # print name, tm
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

                    proc = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)
                    is_proc_opened = True

                    proc.stdin.write(img.tostring())

                    it = 0
                    while it < 200:
                        logging.debug('Before second movement detected')
                        ret, img = cap.read()
                        logging.debug('Second movement detected')

                        if ret:
                            img, status, filtered_mask = detect_movement(f_kernel, img)

                            proc.stdin.write(img.tostring())  # img

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

