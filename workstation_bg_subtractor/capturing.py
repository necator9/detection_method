import detection_logging
import threading
import cv2
import conf
import time
import numpy as np
import glob
import os

CAPTURING_LOG = detection_logging.create_log("capturing.log", "CAPTURING THREAD")


class VirtualCamera(threading.Thread):
    def __init__(self, stop_ev):
        super(VirtualCamera, self).__init__()
        self.stop_event = stop_ev
        self.check_dir()

    def run(self):
        i = 0

        CAPTURING_LOG.info("Virtual camera has started")
        imgs_in_dir = (len(glob.glob(os.path.join(conf.IN_DIR, "*.jpeg")))) - 1
        CAPTURING_LOG.info("Files in directory: {}".format(imgs_in_dir))

        while i < imgs_in_dir and self.stop_event.is_set():
            img_buff = ImgBuff()
            path_to_img = glob.glob(os.path.join(conf.IN_DIR, "img_{}_*.jpeg".format(conf.COUNTER)))[0]
            img_buff.image = cv2.imread(path_to_img)
            img_buff.id = i
            CAPTURING_LOG.info("Image {} has been captured".format(i))
            conf.IMG_BUFF = img_buff
            time.sleep(0.2)
            i += 1

        self.quit()

    def check_dir(self):
        if not os.path.isdir(conf.IN_DIR):
            CAPTURING_LOG.error("INPUT directory does not exists. Path: {}".format(conf.IN_DIR))
            time.sleep(2)
            exit(1)

    def quit(self):
        self.stop_event.clear()


class ImgBuff(object):
    def __init__(self):
        self.image = np.dtype('uint8')
        self.processed = bool()
        self.inserted = bool()

        self.id = int()  # For debug


class Camera(threading.Thread):
    def __init__(self, stop_ev):
        super(Camera, self).__init__()
        self.stop_event = stop_ev
        self.camera = cv2.VideoCapture(conf.IN_DEVICE)  # Initialize the camera capture object

    # Main thread routine

    def run(self):
        CAPTURING_LOG.info("Camera has started capturing")
        self.cam_setup()
        i = 0
        while self.stop_event.is_set():
            start_time = time.time()

            read_ok, img = self.camera.read()

            if not read_ok:
                CAPTURING_LOG.error("Capturing failed")

                break

            CAPTURING_LOG.debug("Image {} is captured".format(i))

            img_buff = ImgBuff()

            img_buff.image = img
            img_buff.id = i
            img_buff.inserted = True

            conf.IMG_BUFF = img_buff

            processing_t = time.time() - start_time
            CAPTURING_LOG.debug("Image shooting takes {}s".format(processing_t))

            i += 1

        self.quit()

    # Camera configuration in accordance to OpenCV version

    def cam_setup(self):
        # Check on successful camera initialization
        if not self.camera.isOpened():
            CAPTURING_LOG.error("Cannot initialize camera")

            self.quit()

        # Initial camera configuration
        # cv_version = int(cv2.__version__.split(".")[0])
        #
        # if cv_version == 3:
        #     self.camera.set(3, conf.ORIG_IMG_RES[0])
        #
        #     self.camera.set(4, conf.ORIG_IMG_RES[1])
        #
        #     self.camera.set(5, config.FPS)
        #
        # if cv_version == 2:
        #
        #     self.camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, config.ORIG_IMG_RES[0])
        #
        #     self.camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, config.ORIG_IMG_RES[1])
        #
        #     self.camera.set(cv2.cv.CV_CAP_PROP_FPS, config.FPS)

    # Stop and quit the thread operation.

    def quit(self):
        self.stop_event.clear()
        self.camera.release()
        CAPTURING_LOG.info("Camera has quit")
