# Camera parameters
DEVICE = "/dev/video0"          # Path to input camera device or video file

# Detector parameters
PROC_IMG_RES = (320, 180)        # Width and height of an image for processing
F_KERNEL_SIZE = (5, 5)          # Size of elliptical filtering kernel in pixels
D_OBJ_SIZE = 0               # Detection threshold. Minimal obj size to be detected
MARGIN = (0, 0)
COEFF_RANGE = (1500, 12000)

# Global variables for internal usage
MOTION_STATUS = False            # Initial status of motion

IMG_IN_DIR = "/home/ivan/test_ir/origin/05.10.17_5"
IMG_OUT_DIR = "/home/ivan/test_ir/detection/05.10.17_5"


