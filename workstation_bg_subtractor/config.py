# Detector parameters
PROC_IMG_RES = [320, 180]        # Width and height of an image for processing
F_KERNEL_SIZE = (5, 5)          # Size of elliptical filtering kernel in pixels
DILATE_ITERATIONS = 1
MARGIN = (0, 0)
COEFF_RANGE = (1200, 12000)
EXTENT_THRESHOLD = 0.5
BRIGHTNESS_THRESHOLD = 0.2
X_MARGIN = 0


# Global variables for internal usage
MOTION_STATUS = bool()

IN_DIR = "/home/ivan/test_ir/origin/05.10.17/3"
OUT_DIR = "/home/ivan/test_ir"
COUNTER = int()
IMG_IN_DIR = int()

SAVE_IMG = False
WRITE_TO_DB = True
SHOW_IMG = False
# WRITE_TO_CSV = False
WRITE_TO_PICKLE = False




