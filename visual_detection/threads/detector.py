import config
from config import *

logger = logging.getLogger(__name__)


def detect(stop_ev, q):

    logger.info("Detector started")

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    while stop_ev.is_set():

        start_time = time.time()

        while stop_ev.is_set():
            if len(config.img_buff) == 0:
                logging.warning("No available images in buffer")
                time.sleep(1)
            else:
                img_proc = config.img_buff
                break

        (rects, weights) = hog.detectMultiScale(img_proc, winStride=(8, 8), padding=(8, 8), scale=1.06)
        if len(rects) == 0:
            q.put(None)
        else:
            q.put(rects)

        tm = (time.time() - start_time)
        logger.debug("Image processing takes: %s s", tm)

        if len(config.mean_t_detector) < 300:
            config.mean_t_detector.append(tm)

        # draw the bounding boxes
        for (x, y, w, h) in rects:
            cv2.rectangle(img_proc, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("my_window", img_proc)
        cv2.waitKey(1)

    logger.info("Detector finished")