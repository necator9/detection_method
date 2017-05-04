import config
from config import *

logger = logging.getLogger(__name__)


def cam_setup(camera, width, height, fps):
    version = int(cv2.__version__.split(".")[0])
    if version == 3:
        camera.set(3, width)
        camera.set(4, height)
        camera.set(5, fps)
    if version == 2:
        camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)
        camera.set(cv2.cv.CV_CAP_PROP_FPS, fps)


def capture(stop_ev):

    logger.info("Grabber started")

    camera = cv2.VideoCapture(0)  # Initialize the camera capture object
    # camera = cv2.VideoCapture("/home/pi/out.m4v")
    # camera = cv2.VideoCapture("/home/ivan/out.m4v")
    if not camera.isOpened():  # Check on successful camera initialization
        logger.error("Cannot initialize camera object")
        os.system(config.command)
        stop_ev.clear()
        sys.exit(-1)

    cam_setup(camera, 320, 240, 7)  # Initial camera configuration: function(object, width, height, fps)

    while stop_ev.is_set():
        start_time = time.time()
        logger.debug("Taking image...")
        ret, config.img_buff = camera.read()
        tm = time.time() - start_time

        if len(config.mean_t_grabber) < 300:
            config.mean_t_grabber.append(tm)

        logger.debug("Image shooting takes %s s", tm)
        # time.sleep(0.2)        # remove!!!!!!!!!!!!!!!!!!!!!!
    camera.release()

    logger.info("Grabber finished")
