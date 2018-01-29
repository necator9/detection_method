# Camera parameters
DEVICE = "/dev/video0"          # Path to input camera device or video file

# Detector parameters
PROC_IMG_RES = [320, 180]        # Width and height of an image for processing
F_KERNEL_SIZE = (5, 5)          # Size of elliptical filtering kernel in pixels
DILATE_ITERATIONS = 3
MARGIN = (0, 0)
COEFF_RANGE = (1300, 12000)
EXTENT_THRESHOLD = 0.6
X_MARGIN = 0


# Global variables for internal usage
MOTION_STATUS = bool()

IN_DIR = "/home/ivan/test_ir/origin/25.10.17/2/"
OUT_DIR = "/home/ivan/test_ir/detection/25.10.17/test/"

SAVE_IMG = False
WRITE_TO_DB = False
SHOW_IMG = False
WRITE_TO_CSV = False

COUNTER = int()
IMG_IN_DIR = int()


