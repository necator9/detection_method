# Camera parameters
DEVICE = "/dev/video0"          # Path to input camera device or video file

# Detector parameters
PROC_IMG_RES = [320, 180]        # Width and height of an image for processing
F_KERNEL_SIZE = (5, 5)          # Size of elliptical filtering kernel in pixels
MARGIN = (0, 0)
COEFF_RANGE = (1700, 12000)
EXTENT_THRESHOLD = 0.5

# Global variables for internal usage
MOTION_STATUS = bool()

IN_DIR = "/home/ivan/test_ir/origin/05.10.17/4/"
OUT_DIR = "/home/ivan/test_ir/detection/05.10.17/4/"

SAVE_IMG = False
WRITE_TO_DB = False
SHOW_IMG = True

COUNTER = int()
IMG_IN_DIR = int()


