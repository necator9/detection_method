from collections import deque

# Shared vars
IMG_BUFF = []                   # Grabber writes into the buff/detector reads from

# Camera parameters
DEVICE = "/dev/video0"          # Path to input camera device or video file

ORIG_IMG_RES = (320, 240)       # Width and height of an image to capture
FPS = 7                         # Capturing frequency (frames per second)

# Detector parameters
PROC_IMG_RES = (100, 70)        # Width and height of an image for processing
F_KERNEL_SIZE = (3, 3)          # Size of elliptical filtering kernel in pixels
D_OBJ_SIZE = 100                # Detection threshold. Minimal obj size to be detected

# Interactive parameters
IMG_SAVE = True                 # Save frames where motion is detected
UI = True                       # Display user interface
SYNC = True

# Sync parameters
BBB_SYNC_DIR = "share/"                       # Path to synchronizing directory on Beaglebone Black
BBB_IMG_DIR = "share/img/"                    # Path to image saving directory on Beaglebone Black
W_SYNC_DIR = "~/share_BBB/"                   # Path to synchronizing directory on workstation
W_USER_IP = "ivan@192.168.100.119"                  # Workstation username@ip
W_PORT = "2122"                                     # Workstation ssh port
COMMAND = "rsync -avzhe 'ssh -p %s' --delete %s %s:%s" % (W_PORT, BBB_SYNC_DIR, W_USER_IP, W_SYNC_DIR)

# Global variables for internal usage
MOTION_STATUS = False            # Initial status of motion
T_DETECTOR = deque(maxlen=30)    # Times of detection duration are stored here
T_GRABBER = deque(maxlen=30)     # Times of capturing duration are stored here
# Avoid operations with empty deque
T_DETECTOR.append(0)
T_GRABBER.append(0)




