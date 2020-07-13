import conf
from capturing import StartAppError
import sl_sensor_connect
import cv2
import logging

logger = logging.getLogger('detect.pre_processing')


class PreprocessImg(object):
    def __init__(self):
        # Background subtraction parameters
        self.shadows = conf.SHADOWS
        self.history = conf.HISTORY
        self.bg_thr = conf.BG_THR
        self.bgs_method = self.create_bgs(conf.BGS_METHOD)

        self.f_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.clahe_adjust = cv2.createCLAHE(clipLimit=conf.CLAHE_LIMIT, tileGridSize=(8, 8))
        self.set_ratio_done = bool()

        self.sl_app_conn = sl_sensor_connect.SlSensor(send_port=65433, recv_port=65434)

    def create_bgs(self, flag):
        if flag == 'MOG2':
            bgs_method = cv2.createBackgroundSubtractorMOG2(detectShadows=self.shadows, history=self.history,
                                                            varThreshold=self.bg_thr)
        elif flag == 'KNN':
            bgs_method = cv2.createBackgroundSubtractorKNN(detectShadows=self.shadows, history=self.history)
        else:
            raise StartAppError

        return bgs_method

    def apply(self, orig_img):
        orig_img = cv2.resize(orig_img, conf.RES, interpolation=cv2.INTER_NEAREST)
        # Update processing resolution according to one after resize (i.e. not correct res. is chosen by user)
        self.set_ratio(orig_img)

        orig_img = self.clahe_adjust.apply(orig_img)

        # Create new background model when lamp is switched on or off
        data = self.sl_app_conn.recieve()
        if data == 'Lamp_ON' or data == 'Lamp_OFF':
            self.bgs_method = self.create_bgs(conf.BGS_METHOD)
            logger.info('Signal from SL_app received. The background model {} updated'.format(conf.BGS_METHOD))

        mog_mask = self.bgs_method.apply(orig_img)

        filtered_mask = cv2.morphologyEx(mog_mask, cv2.MORPH_OPEN, self.f_kernel)

        _, filled_mask = cv2.threshold(filtered_mask, 170, 255, cv2.THRESH_BINARY)

        if conf.DILATE_ITERATIONS:
            filled_mask = cv2.dilate(filled_mask, None, iterations=conf.DILATE_ITERATIONS)

        return orig_img, mog_mask, filtered_mask, filled_mask

    def set_ratio(self, img):
        if not self.set_ratio_done:
            self.set_ratio_done = True
            actual_w, actual_h = img.shape[:2][1], img.shape[:2][0]
            logger.info("Processing resolution: {}x{}".format(actual_w, actual_h))

            if conf.RES[0] != actual_w or conf.RES[1] != actual_h:
                conf.RES[0], conf.RES[1] = actual_w, actual_h
                logger.info("Processing resolution updated: {}x{}".format(actual_w, actual_h))
                # global PINHOLE_CAM
                # PINHOLE_CAM = init_pcm()