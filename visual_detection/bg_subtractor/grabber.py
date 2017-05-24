import config
from config import *

logger = logging.getLogger(__name__)


def cam_setup(camera):                      # Camera configuration in accordance to OpenCV version
    cv_version = int(cv2.__version__.split(".")[0])

    if cv_version == 3:
        camera.set(3, config.width)
        camera.set(4, config.height)
        camera.set(5, config.fps)

    if cv_version == 2:
        camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, config.width)
        camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, config.height)
        camera.set(cv2.cv.CV_CAP_PROP_FPS, config.fps)


def capture(stop_ev):
    logger.info("Grabber started")

    camera = cv2.VideoCapture(config.dev)   # Initialize the camera capture object

    if not camera.isOpened():               # Check on successful camera initialization
        logger.error("Cannot initialize camera object")
        stop_ev.clear()
        sys.exit(-1)

    cam_setup(camera)                       # Initial camera configuration

    while stop_ev.is_set():
        start_time = time.time()
        logger.debug("Taking image...")

        ret, img = camera.read()            # Getting of image into img
        config.img_buff = imutils.resize(img, width=100)

        if len(config.t_grabber) < config.st_window:
            config.t_grabber.append(time.time() - start_time)

        logger.debug("Image shooting takes %s s", time.time() - start_time)

    camera.release()

    logger.info("Grabber finished")
