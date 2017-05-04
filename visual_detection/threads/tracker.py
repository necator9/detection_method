import collections
from config import *

logger = logging.getLogger(__name__)


def calc(deq, shift_min, shift_max):
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
                if shift_x and shift_y == 0:
                    print "kek"
                if (shift_x < shift_max and shift_y < shift_max) \
                        and (shift_x > shift_min or shift_y > shift_min):
                    return True


def track(stop_ev, rects, shift_min=0, shift_max=25):

    logger.info("Tracker started")

    deq = collections.deque(maxlen=5)
    counter = 0
    while stop_ev.is_set():
        try:
            val = rects.get(timeout=1)
        except Queue.Empty:
            logger.warning("No values in queue")
            continue

        deq.append(val)
        detection = calc(deq, shift_min, shift_max)
        if detection:
            logger.info("human movement detected")
            counter += 1

    logger.info("Tracker finished")
    logger.info(counter)






