from collections import deque

# Shared vars
IMG_BUFF = []                   # Grabber writes into the buff/detector reads from

# Camera parameters
DEVICE = "/dev/video0"             # Path to input camera device or video file
# dev = "/home/pi/out.m4v"
# dev = "/home/ivan/out.m4v"
IMG_WIDTH = 320                 # Width of an image to capture
IMG_HEIGHT = 240                # Height of an image to capture
FPS = 7                         # Capturing frequency (frames per second)

# Image parameters to resize, compress and save images
IMG_WIDTH_SAVE = 100            # Width of an image to capture
IMG_HEIGHT_SAVE = 70            # Height of an image to capture

# Detector parameters
FILTERED_OBJ_SIZE = (3, 3)      # Size of elliptical filtering kernel in pixels
DETECTED_OBJ_SIZE = 30          # Detection threshold. Minimal obj size to be detected

# Interactive parameters
IMG_SAVE = True                 # Save frames where motion is detected
UI = True                       # Display user interface
SYNC_DIR = True

# Sync parameters
BBB_SYNC_DIRECTORY = "share/"
BBB_IMG_DIRECTORY = "share/img/"
WORKSTATION_SYNC_DIRECTORY = "~/share_BBB/"
WORKSTATION_USER_IP = "ivan@192.168.100.119"
WORKSTATION_PORT = "2122"
COMMAND = "rsync -avzhe 'ssh -p %s' --delete %s %s:%s" % \
          (WORKSTATION_PORT, BBB_SYNC_DIRECTORY, WORKSTATION_USER_IP, WORKSTATION_SYNC_DIRECTORY)

# Global variables for internal usage
MOTION_STATUS = False            # Initial status of motion
T_DETECTOR = deque(maxlen=30)    # Times of detection duration are stored here
T_GRABBER = deque(maxlen=30)     # Times of capturing duration are stored here
# Avoid operations with empty deque
T_DETECTOR.append(0)
T_GRABBER.append(0)




