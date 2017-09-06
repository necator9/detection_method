
# Shared vars
IMG_BUFF = []                   # Grabber writes into the buff/detector reads from

# Camera parameters
DEVICE = "/dev/video0"          # Path to input camera device or video file

ORIG_IMG_RES = (640, 480)       # Width and height of an image to capture
FPS = 30                        # Capturing frequency (frames per second)

# Detector parameters
PROC_IMG_RES = (640, 480)        # Width and height of an image for processing
F_KERNEL_SIZE = (3, 3)          # Size of elliptical filtering kernel in pixels
D_OBJ_SIZE = 0                # Detection threshold. Minimal obj size to be detected


# Global variables for internal usage
MOTION_STATUS = False            # Initial status of motion


IMG_SAVE = False
IMG_DISPLAY = True




