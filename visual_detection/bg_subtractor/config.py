# Shared vars
IMG_BUFF = []                   # Grabber writes in the var/detector reads from the var

# Camera parameters
DEV = 0                         # Number of a device (camera) to use (/dev/video<number>)
# dev = "/home/pi/out.m4v"
# dev = "/home/ivan/out.m4v"
IMG_WIDTH = 320                     # Width of an image to capture
IMG_HEIGHT = 240                    # Height of an image to capture
FPS = 7                         # Capturing frequency (frames per second)

# Detector parameters
FILTERED_OBJ_SIZE = (10, 10)             # Size of elliptical filtering kernel in pixels
DETECTED_OBJ_SIZE = 900                 # Detection threshold. Minimal obj size to be detected

# Statistic parameters
IMG_SAVE = True
COMMAND = "rsync -avzhe 'ssh -p 2122' --delete ../share/ ivan@192.168.100.119:~/share_BBB/"
PATH_TO_SHARE = "../share/"

ST_WINDOW = 300                 # Window size for timing calculation
T_DETECTOR = []                 # Times of detection duration are stored here
T_GRABBER = []                  # Times of capturing duration are stored here
DETECTIONS_AMOUNT = 0           # Amount of detection during execution (the value is incremented)




