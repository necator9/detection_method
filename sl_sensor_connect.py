#!/usr/bin/env python3

import socket
import select
import time


class SlSensor(object):
    """
    Communication between SmartLighting and detection algorithm is defined by this class.
    Each of the applications uses 2 sockets: one for sending and one for receiving.
    Thus, the sending port in one application is the receiving port in another, and vice versa.
    """

    def __init__(self, send_port, recv_port, host='127.0.0.1'):
        self.host = host
        self.send_port = send_port
        self.recv_port = recv_port

        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)  # Set size of buf to minimal value to
        # limit the length of a queue that may be required when applications' iteration times are different

        self.sock_recv.bind((self.host, self.recv_port))

    def recieve(self):
        """
        Non blocking recieve to avoid try-except by timeout.
        :return:
        """
        ready = select.select([self.sock_recv], [], [], 0)
        if ready[0]:
            data, _ = self.sock_recv.recvfrom(16)
            return data.decode()

    def send(self, data):
        self.sock_send.sendto(str(data).encode(), (self.host, self.send_port))

    def quit(self):
        self.sock_send.close()
        self.sock_recv.close()


if __name__ == '__main__':
    sl_app = SlSensor(send_port=65434, recv_port=65433)
    sensor_app = SlSensor(send_port=65433, recv_port=65434)

    try:
        while True:
            sl_app.send('The lamp is ON')
            print('Sensor received: {}'.format(sensor_app.recieve()))
            time.sleep(2)
            sl_app.send('The lamp is OFF')
            print('Sensor received: {}'.format(sensor_app.recieve()))
            time.sleep(2)
            sensor_app.send('Object_detected')
            print('SL_app  received: {}'.format(sl_app.recieve()))
            time.sleep(2)

    except KeyboardInterrupt:
        pass

    finally:
        sl_app.quit()
        sensor_app.quit()
