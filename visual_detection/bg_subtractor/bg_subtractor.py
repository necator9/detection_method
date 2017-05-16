import cv2


def cam_setup(camera, width, height, fps):                      # Camera configuration in accordance to OpenCV version
    cv_version = int(cv2.__version__.split(".")[0])

    if cv_version == 3:
        camera.set(3, width)
        camera.set(4, height)
        camera.set(5, fps)

    if cv_version == 2:
        camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)
        camera.set(cv2.cv.CV_CAP_PROP_FPS, fps)


mog = cv2.createBackgroundSubtractorMOG2()
# mog = cv2.bgsegm.createBackgroundSubtractorMOG()
# mog = cv2.bgsegm.createBackgroundSubtractorGMG()

camera = cv2.VideoCapture(0)
cam_setup(camera, 320, 240, 7)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

while True:
    ret_val, image = camera.read()
    mask = mog.apply(image)

    # cv2.imshow("mask", mask)
    # cv2.waitKey(1)

    filtered = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("opening", filtered)
    # cv2.waitKey(1)

    filled = cv2.dilate(filtered, None, iterations=8)

    _, cnts, hier = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for arr in cnts:
        if cv2.contourArea(arr) < 1000:
            continue
        else:
            print "Motion detected"

        # (x, y, w, h) = cv2.boundingRect(arr)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # cv2.imshow("result", image)
    # cv2.waitKey(1)
