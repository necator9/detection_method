import cv2
import time

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(1)
    cam.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 320)
    cam.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 240)
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
        time.sleep(1)
    cv2.destroyAllWindows()

def main():
    show_webcam(mirror=True)

if __name__ == '__main__':
    main()