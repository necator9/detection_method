# Detector parameters
PROC_IMG_RES = [320, 180]        # Width and height of an image for processing
F_KERNEL_SIZE = (5, 5)          # Size of elliptical filtering kernel in pixels
DILATE_ITERATIONS = 1
MARGIN = (0, 0)
COEFF_RANGE = (1200, 12000)
EXTENT_THRESHOLD = 0.5
BRIGHTNESS_THRESHOLD = 0.2
X_MARGIN = 0

OUT_DIR = "/root/test"
PATH_TO_LOGS = "/root/"

# PATH_TO_LOGS = "/home/ivan/test_ir/"
# OUT_DIR = "/home/ivan/test_ir/test"

# Camera parameters
IN_DEVICE = "/dev/video0"
ORIG_IMG_RES = (320, 240)
FPS = 7

# Virtual camera parameters
VIRTUAL_CAMERA = False
IN_DIR = "/home/ivan/test_ir/origin/05.10.17/3"


COUNTER = int()

SAVE_IMG = False
WRITE_TO_DB = False
WRITE_TO_PICKLE = False

TIMERS = False
SHOW_LOGS = False
LOG_LEVEL = "INFO"


# Global variables for internal usage
MOTION_STATUS = bool()









# Outdated functional
# SHOW_IMG = False
# WRITE_TO_CSV = False




