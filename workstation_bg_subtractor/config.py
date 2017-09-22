
# Camera parameters
DEVICE = "/dev/video0"          # Path to input camera device or video file

# Detector parameters
PROC_IMG_RES = (320, 240)        # Width and height of an image for processing
F_KERNEL_SIZE = (4, 4)          # Size of elliptical filtering kernel in pixels
D_OBJ_SIZE = 1500               # Detection threshold. Minimal obj size to be detected


# Global variables for internal usage
MOTION_STATUS = False            # Initial status of motion

IMG_IN_DIR = "/home/ivan/test_ir/share/img/"
IMG_OUT_DIR = "/home/ivan/test_ir/detection/"


