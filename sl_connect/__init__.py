#!/usr/bin/env python3
# Compatible to python 2.7

# Created by Ivan Matveev at 04.08.20
# E-mail: ivan.matveev@hs-anhalt.de

# Interconnection between Smartlighting application (running as a server) and multiple detection algorithm or lamps
# instances (running as clients). The module contains classes to implement server and client functionality.
#  A simple interface dedicated to applications' needs is provided upon the transportation logic.

import socket
import select
import logging

# Set up logging for detection application
logger = logging.getLogger('detect.sl_connect')


class SocketInterface(object):
    """
    Server sends and receives messages to/from clients.
    """
    def __init__(self, server_port, clients_ports):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Set size of buf to minimal value to limit the length of a queue (UDP buffer) that may be required when
        # applications' iteration speeds are different or server disabled temporary. In current implementation
        # 1024 bytes corresponds to 4 messages max in UDP buffer - not relevant messages are discarded.
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('127.0.0.1', server_port))
        self.clients = [('127.0.0.1', port) for port in clients_ports]

    def receive(self):
        ready = select.select([self.sock], [], [], 0)  # Non-blocking receive to avoid try-except by timeout.
        if ready[0]:
            data, addr = self.sock.recvfrom(16)
            data = data.decode()
            logger.debug('Msg received: {}'.format(data))
            return data

    def send(self, data):
        logger.debug('Sending {} to clients: {}'.format(data, self.clients))
        for client_addr in self.clients:  # Send to all clients in pool
            self.sock.sendto(str(data).encode(), client_addr)

    def quit(self):
        self.sock.close()


class DetectAlgorithmConn(SocketInterface):
    """
    Abstract class to be used in SmartLighting application. Provides narrowed interface for interaction with the
    detection algorithm
    """
    def __init__(self, *args, **kwargs):
        super(DetectAlgorithmConn, self).__init__(*args, **kwargs)

    def check_detect_status(self):
        """
        Called in each cycle of the main program
        """
        return True if self.receive() == 'DETECTED' else False

    def lamp_on(self):
        self.send('LAMP_ON')

    def lamp_off(self):
        self.send('LAMP_OFF')


class SlAppConnSensor(SocketInterface):
    """
    Abstract class to be used in detection algorithm. Provides narrowed interface for interaction with the
    SmartLighting application
    """
    def __init__(self, *args, **kwargs):
        super(SlAppConnSensor, self).__init__(*args, **kwargs)

    def check_lamp_status(self):
        """
        Called in each cycle of the main program
        """
        data = self.receive()
        return True if data == 'LAMP_ON' or data == 'LAMP_OFF' else False

    def switch_on_lamp(self):
        self.send('DETECTED')
