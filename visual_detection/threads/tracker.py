import collections
import config
from config import *

logger = logging.getLogger(__name__)


def calc(deq):
    if len(deq) < 2:
        logger.debug("Length of deq is less then 2")
        return

    for k in range(len(deq) - 1):
        if deq[k] is None:
            continue
        if deq[-1] is None:
            return

        for x, y, w, h in deq[k]:
            for i in range(len(deq[-1])):
                shift_x = abs(deq[-1][i][0] - x)
                shift_y = abs(deq[-1][i][1] - y)
                if (shift_x < config.shift_max and shift_y < config.shift_max) \
                        and (shift_x > config.shift_min or shift_y > config.shift_min):
                    return True
                else:
                    return False


def track(stop_ev, coordinates):
    logger.info("Tracker started")

    deq = collections.deque(maxlen=config.deq_len)

    while stop_ev.is_set():
        try:
            val = coordinates.get(timeout=1)
        except Queue.Empty:
            logger.warning("No values in queue")
            continue

        deq.append(val)
        detection = calc(deq)
        if detection:
            logger.info("human movement detected")
            config.detections_amount += 1

    logger.info("Tracker finished")
    logger.info("Amount of detections: %s" % config.detections_amount)






