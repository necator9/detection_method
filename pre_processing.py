import cv2
import logging

logger = logging.getLogger('detect.pre_processing')


class PreprocessImg(object):
    def __init__(self, config):
        # Background subtraction parameters
        self.bgs_method_name = config['bgs_method']['name']
        self.bgs_parameters = config['bgs_method']['parameters']

        self.bgs_map = {'MOG2': cv2.createBackgroundSubtractorMOG2, 'KNN': cv2.createBackgroundSubtractorKNN,
                        'CNT': cv2.bgsegm.createBackgroundSubtractorCNT}
        self.bgs_method = self.bgs_map[self.bgs_method_name](*self.bgs_parameters)

        self.f_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.clahe_adjust = cv2.createCLAHE(clipLimit=config['clahe_limit'], tileGridSize=(8, 8))
        self.dilate_iterations = int(config['dilate_it'])
        self.resolution = tuple(config['resolution'])

    def apply(self, orig_img, lamp_status):
        orig_img = cv2.resize(orig_img, self.resolution, interpolation=cv2.INTER_NEAREST)
        orig_img = self.clahe_adjust.apply(orig_img)

        # Create new background model when lamp is switched on or off
        if lamp_status:
            self.bgs_method = self.bgs_map[self.bgs_method_name](*self.bgs_parameters)
            logger.info('Signal from SL_app received. The background model {} updated'.format(self.bgs_method_name))

        mog_mask = self.bgs_method.apply(orig_img)
        filtered_mask = cv2.morphologyEx(mog_mask, cv2.MORPH_OPEN, self.f_kernel)
        _, filled_mask = cv2.threshold(filtered_mask, 170, 255, cv2.THRESH_BINARY)

        if self.dilate_iterations:
            filled_mask = cv2.dilate(filled_mask, None, iterations=self.dilate_iterations)

        return orig_img, mog_mask, filtered_mask, filled_mask
