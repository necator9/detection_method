import conf
from capturing import StartAppError
import imutils
import cv2
# from detection import  DETECTION_LOG


class PreprocessImg(object):
    def __init__(self):
        if conf.BGS_METHOD == 'MOG2':
            self.bgs_method = cv2.createBackgroundSubtractorMOG2(detectShadows=True, history=1500, varThreshold=16)
        elif conf.BGS_METHOD == 'KNN':
            self.bgs_method = cv2.createBackgroundSubtractorKNN(detectShadows=True, history=1500)
        else:
            raise StartAppError

        self.f_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.clahe_adjust = cv2.createCLAHE(clipLimit=conf.CLAHE_LIMIT, tileGridSize=(8, 8))
        self.set_ratio_done = bool()

    def process(self, orig_img):
        orig_img = imutils.resize(orig_img, height=conf.IMG_RES[1])
        # Update processing resolution according to one after resize (i.e. not correct res. is chosen by user)
        self.set_ratio(orig_img)

        if not conf.COLOR:
            orig_img = self.clahe_adjust.apply(orig_img)
        # orig_img = cv2.blur(orig_img, (5, 5))

        mog_mask = self.bgs_method.apply(orig_img)
        # filtered_mask = mog_mask
        filtered_mask = cv2.morphologyEx(mog_mask, cv2.MORPH_OPEN, self.f_kernel)
        # filtered_mask = cv2.blur(filtered_mask, (3, 3))

        _, filled_mask = cv2.threshold(filtered_mask, 170, 255, cv2.THRESH_BINARY)

        # filtered_mask = cv2.morphologyEx(mog_mask, cv2.MORPH_OPEN, self.f_kernel)
        if conf.DILATE_ITERATIONS:
            filled_mask = cv2.dilate(filled_mask, None, iterations=conf.DILATE_ITERATIONS)
        # filled_mask = cv2.blur(filtered_mask, (3, 3))
        # filled_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, self.f1_kernel)

        return orig_img, mog_mask, filtered_mask, filled_mask

    def set_ratio(self, img):
        if not self.set_ratio_done:
            self.set_ratio_done = True
            actual_w, actual_h = img.shape[:2][1], img.shape[:2][0]
            # DETECTION_LOG.info("Processing resolution: {}x{}".format(actual_w, actual_h))

            if conf.IMG_RES[0] != actual_w or conf.IMG_RES[1] != actual_h:
                conf.IMG_RES[0], conf.IMG_RES[1] = actual_w, actual_h
                # global PINHOLE_CAM
                # PINHOLE_CAM = init_pcm()