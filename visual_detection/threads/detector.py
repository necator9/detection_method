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


def detect(stop_ev, q):
    logger.info("Detector started")

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    while stop_ev.is_set():
        start_time = time.time()

        if check_on_buffer():
            img = config.img_buff
        else:
            continue

        (coordinates, weights) = hog.detectMultiScale(img,
                                                      winStride=config.winStr,
                                                      padding=config.pad,
                                                      scale=config.scale)
        if len(coordinates) == 0:
            q.put(None)
        else:
            q.put(coordinates)

        if len(config.t_detector) < 300:
            config.t_detector.append(time.time() - start_time)

        logger.debug("Image processing takes: %s s", time.time() - start_time)

        # Draw the bounding boxes
        for (x, y, w, h) in coordinates:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("my_window", img)
        cv2.waitKey(1)

    logger.info("Detector finished")
