#!/usr/bin/python
import socket
import cv2
import numpy
import conf


def recv_all(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf:

            return None

        buf += newbuf
        count -= len(newbuf)

    return buf


TCP_IP = conf.SERVER_TCP_IP
TCP_PORT = 5001

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(True)
conn, addr = s.accept()
try:
    while True:
        length = recv_all(conn, 16)
        if length is None:

            break

        stringData = recv_all(conn, int(length))
        data = numpy.fromstring(stringData, dtype='uint8')

        dec_img = cv2.imdecode(data, 1)
        cv2.imshow('SERVER', dec_img)
        cv2.waitKey(1)
except KeyboardInterrupt:
    print "Keyboard interrupt"

s.close()

