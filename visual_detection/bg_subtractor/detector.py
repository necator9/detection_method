import config
from config import *

logger = logging.getLogger(__name__)


def check_on_buffer():
    if len(config.img_buff) == 0:
        logging.warning("No available images in buffer")
        time.sleep(1)
        return False
    else:
        return True


def detect(stop_ev):
    logger.info("Detector started")

    mog = cv2.createBackgroundSubtractorMOG2()
    #mog = cv2.bgsegm.createBackgroundSubtractorMOG()
    #mog = cv2.bgsegm.createBackgroundSubtractorGMG()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.fObjSize)

    i = 0
    while stop_ev.is_set():
        start_time = time.time()

        if check_on_buffer():
            img = config.img_buff
        else:
            continue

        mask = mog.apply(img)
        filtered = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        filled = cv2.dilate(filtered, None, iterations=8)

        _, cnts, hier = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for arr in cnts:
            if cv2.contourArea(arr) > config.dObjSize:
                logging.info("Motion detected")
                if config.img_save_flag:
                    (x, y, w, h) = cv2.boundingRect(arr)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.imwrite("../share/img/img_%s.jpeg" % i, img)
            else:
                continue

        if len(config.t_detector) < 300:
            config.t_detector.append(time.time() - start_time)
        i += 1
        logger.info("Image processing takes: %s s", time.time() - start_time)

    logger.info("Detector finished")
