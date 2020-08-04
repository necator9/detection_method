#!/usr/bin/env python3

import socket
import select
import time
import logging

# Set up logging for detection application
logger = logging.getLogger('detect.sl_connect')


class SensorServer(object):
    """
    Communication between SmartLighting and detection algorithm is defined by this class.
    """
    def __init__(self, server_addr=('', 65433), max_clients=2):
        self.sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)  # Set size of buf to minimal value to
        # limit the length of a queue that may be required when applications' iteration times are different
        self.sock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock_recv.bind(server_addr)
        self.max_clients = max_clients
        self.clients = dict()

    def receive(self):
        ready = select.select([self.sock_recv], [], [], 0)  # Non-blocking receive to avoid try-except by timeout.
        if ready[0]:
            data, addr = self.sock_recv.recvfrom(16)
            data = data.decode()
            self.clients[addr] = time.time()
            logger.debug('Msg received: {}'.format(data))
            if len(self.clients) > self.max_clients:
                self.remove_oldest_client()
            else:
                return data

    def remove_oldest_client(self):
        oldest_client = min(self.clients, key=self.clients.get)
        logger.info('Number of clients exceeds max allowed ({}>{}). Removing the oldest client: {}'.
                    format(len(self.clients), self.max_clients, oldest_client))
        del self.clients[oldest_client]

    def send(self, data):
        logger.debug('Sending {} to clients: {}'.format(data, self.clients.keys()))
        for client_addr in self.clients.keys():
            self.sock_recv.sendto(str(data).encode(), client_addr)

    def quit(self):
        self.sock_recv.close()


class SensorClient(object):
    def __init__(self, server_addr=('', 65433)):
        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_addr = server_addr
        self.counter = int()
        self.send('INIT')  # Send msg containing non-keyword content to register client
        logger.info('Notification sent')
        self.receive_call_count = 0

    def receive(self):
        self.maintain_connection()
        ready = select.select([self.sock_send], [], [], 0)
        if ready[0]:
            data, addr = self.sock_send.recvfrom(16)
            return data.decode()

    def maintain_connection(self):
        """
        Sends notification each nth cycle to the server. Designed to reestablish connection when SmartLighting app
        restarted and client's pool erased.
        """
        if self.receive_call_count > 100:
            self.send('M_INIT')
            self.receive_call_count = 0
        self.receive_call_count += 1

    def send(self, data):
        logger.debug('Mgs sent: {}'.format(data))
        self.sock_send.sendto(str(data).encode(), self.server_addr)

    def quit(self):
        self.sock_send.close()


class DetectAlgorithm(SensorServer):
    """
    Abstract class to be used in SmartLighting application. Provides narrowed interface for interaction with the
    detection algorithm
    """
    def __init__(self, *args, **kwargs):
        super(DetectAlgorithm, self).__init__(*args, **kwargs)

    def check_detect_status(self):
        return True if self.receive() == 'LAMP_ON' else False

    def lamp_on_inform(self):
        self.send('SL_LAMP_ON')


class SlApp(SensorClient):
    """
    Abstract class to be used in detection algorithm. Provides narrowed interface for interaction with the
    SmartLighting application
    """
    def __init__(self, *args, **kwargs):
        super(SlApp, self).__init__(*args, **kwargs)

    def check_lamp_status(self):
        return True if self.receive() == 'SL_LAMP_ON' else False

    def switch_on_lamp(self):
        self.send('LAMP_ON')
