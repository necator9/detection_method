#!/usr/bin/python

import numpy
import socket
import capturing
import threading
import detection_logging
import Queue
import time
import cv2


STREAMING_LOG = detection_logging.create_log("streaming.log", "STREAMING CLIENT")

STREAMING_LOG.info("Program has started")

TCP_IP = "192.168.4.8"
TCP_PORT = 5001

sock = socket.socket()
sock.connect((TCP_IP, TCP_PORT))

stop_event = threading.Event()
stop_event.set()

orig_img_q = Queue.Queue(maxsize=1)

capturing_thread = capturing.Camera(orig_img_q, stop_event)

capturing_thread.start()

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
counter = int()

try:
    while stop_event.is_set():
        try:
            image = orig_img_q.get(timeout=2)
            result, imgencode = cv2.imencode('.jpg', image, encode_param)
            data = numpy.array(imgencode)
            stringData = data.tostring()

            sock.send(str(len(stringData)).ljust(16))
            sock.send(stringData)

        except Queue.Empty:
            STREAMING_LOG.info("orig_img_queue is empty, next iteration")

            continue

        STREAMING_LOG.info("{} images were sent".format(counter))

        # time.sleep(1)
        counter += 1

except KeyboardInterrupt:
    STREAMING_LOG.warning("Keyboard Interrupt, threads are going to stop")

capturing_thread.quit()
sock.close()


STREAMING_LOG.info("Program finished")

time.sleep(1)
detection_logging.stop_log_thread()

